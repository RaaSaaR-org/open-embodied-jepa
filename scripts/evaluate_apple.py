"""Supervised dense-image-waypoint development evaluation; final cohort is opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ENTRY_CLOCK = (time.time(), time.monotonic())

from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.contracts import EE_DELTA_GRASP_V0, ContractError  # noqa: E402
from embodied_jepa.training import json_hash  # noqa: E402
from embodied_jepa.waypoint_planning import (  # noqa: E402
    ImageWaypoint,
    StateWaypoint,
    WaypointConfig,
    WaypointController,
)

MODES = (
    "learned",
    "persistence",
    "dynamics_shuffle",
    "hold",
    "random",
    "demo_replay",
    "privileged_rollout",
    "privileged_hybrid",
    "privileged_hybrid_early",
    "privileged_object",
    "scripted_oracle",
)
# TASK-043 demonstration-state-goal MPC. demo_replay is a NON-LEARNED open-loop
# replay of the retrieved TRAIN demonstration; it never calls the world model.
STATE_MODES = ("learned", "dynamics_shuffle", "persistence", "demo_replay")
IMAGE_MODES = MODES[:5]
STATE_BACKEND = "state_goal_sensor_wm_v1"
# TASK-044 NON-LEARNED privileged ceiling: the same state-goal CEM scaffold with exact
# MuJoCo rollouts as its forward model. It runs alone, in development only, and is
# never a learned result (see embodied_jepa.privileged_rollout).
PRIVILEGED_MODE = "privileged_rollout"
CONTROLLER_MODES = ("learned", "persistence", "dynamics_shuffle", PRIVILEGED_MODE)
# TASK-045 time-indexed trajectory tracking over the same retrieved TRAIN demonstration.
TRACKING_MODES = ("learned", "dynamics_shuffle", "persistence")
# Reference frames that move less than this fraction of the demonstration's median
# positive adjacent-frame distance from the last kept frame are dead time.
KEYFRAME_FRACTION = 0.1
STAGES = ("reach", "grasp", "transport", "place", "release")
# TASK-046 NON-LEARNED hybrid phase control under the privileged ceiling: exact-rollout
# trajectory tracking until the measured reference index reaches the handoff row, then
# open-loop replay of the retrieved TRAIN demonstration's own actions from the matched
# frame. The primary arm hands off at the first close row, the secondary arm one
# planning horizon earlier; demo_replay is the same-code open-loop reference.
HYBRID_PRIVILEGED_MODES = ("privileged_hybrid", "privileged_hybrid_early")
HYBRID_MODES = ("privileged_hybrid", "demo_replay", "privileged_hybrid_early")
# TASK-047 wide-jitter development distribution (new resets 45000-45007; the narrow
# 43000-43004 development and 44000-44019 final cohorts are unchanged) with three
# NON-LEARNED arms: the retrieved TRAIN demo_replay, the privileged scripted collector
# policy (feasibility reference) and the object-aware privileged MuJoCo-rollout ceiling.
OBJECT_MODE = "privileged_object"
OBJECT_MODES = ("demo_replay", "scripted_oracle", OBJECT_MODE)
OBJECT_ONLY_MODES = ("scripted_oracle", OBJECT_MODE)
WIDE_COHORT = tuple(range(45000, 45008))
WIDE_JITTER_M = {"object_xy": 0.03, "plate_xy": 0.02}
RESET_CENTERS = {"object_xy": (0.34, -0.18), "plate_xy": (0.49, -0.09)}
OBJECT_WALL_LIMIT = 9000


def validate_attempt_budget(seconds):
    if seconds is not None and (
        isinstance(seconds, bool) or not np.isfinite(seconds) or not 0 < seconds <= 1800
    ):
        raise ValueError("attempt wall budget must lie in (0,1800]seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def make_plan(
    *,
    stage="development",
    seeds=None,
    modes=None,
    max_seconds=600,
    max_steps=1000,
    stride=10,
    dwell=3,
    horizon=4,
    candidates=16,
    iterations=2,
    proposals=True,
    control_timeout=5.0,
    commitment_steps=1,
    attempt_max_seconds=None,
    goal_kind="image",
    goal_stall_limit=None,
):
    if goal_kind == "object":
        return make_object_plan(
            stage=stage,
            seeds=seeds,
            modes=modes,
            max_seconds=max_seconds,
            max_steps=max_steps,
            stride=stride,
            dwell=dwell,
            horizon=horizon,
            candidates=candidates,
            iterations=iterations,
            proposals=proposals,
            control_timeout=control_timeout,
            commitment_steps=commitment_steps,
            attempt_max_seconds=attempt_max_seconds,
            goal_stall_limit=goal_stall_limit,
        )
    if goal_kind not in ("image", "state", "trajectory", "hybrid"):
        raise ValueError("goal kind must be image, state, trajectory, hybrid or object")
    if stage not in ("development", "final"):
        raise ValueError("stage must be development or final")
    cohort = tuple(range(43000, 43005)) if stage == "development" else tuple(range(44000, 44020))
    seeds = list(cohort if seeds is None else seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s not in cohort for s in seeds)
    ):
        raise ValueError("seeds must be unique members of the declared stage cohort")
    if stage == "final" and seeds != list(cohort):
        raise ValueError("final evaluation requires the complete frozen20-reset cohort")
    default_modes = {
        "state": STATE_MODES,
        "trajectory": TRACKING_MODES,
        "hybrid": HYBRID_MODES,
    }.get(goal_kind, IMAGE_MODES)
    modes = list(default_modes if modes is None else modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("unknown or repeated control mode")
    if any(m in OBJECT_ONLY_MODES for m in modes):
        raise ValueError("object-aware ceiling modes require --goal-kind object")
    if goal_kind == "hybrid":
        if any(m not in HYBRID_MODES for m in modes) or stage != "development":
            raise ValueError(
                "hybrid phase control is a non-learned development ceiling: "
                "privileged_hybrid/demo_replay/privileged_hybrid_early only"
            )
    elif any(m in HYBRID_PRIVILEGED_MODES for m in modes):
        raise ValueError("hybrid phase modes require --goal-kind hybrid")
    privileged = PRIVILEGED_MODE in modes
    if privileged and (modes != [PRIVILEGED_MODE] or stage != "development"):
        raise ValueError(
            "the privileged simulator-rollout ceiling runs alone at the development stage; "
            "it is never mixed with learned modes or used for final evaluation"
        )
    if privileged and goal_kind not in ("state", "trajectory"):
        raise ValueError("the privileged ceiling uses the state-goal scaffold only")
    if goal_kind == "image":
        if "demo_replay" in modes:
            raise ValueError("demo_replay requires --goal-kind state demonstration retrieval")
        if goal_stall_limit is not None:
            raise ValueError("goal stall limit is part of the state-goal protocol only")
        wall_limit = 1800
    else:
        allowed = {"state": STATE_MODES, "trajectory": TRACKING_MODES}.get(goal_kind, HYBRID_MODES)
        if any(m not in allowed + (PRIVILEGED_MODE,) for m in modes):
            if goal_kind == "trajectory":
                raise ValueError("trajectory modes: learned/dynamics_shuffle/persistence")
            raise ValueError("state goals support learned/dynamics_shuffle/persistence/demo_replay")
        if proposals:
            raise ValueError("state-goal protocol forbids demonstration action proposals")
        if stride != horizon:
            raise ValueError("state goals must be spaced exactly one planning horizon apart")
        if type(goal_stall_limit) is not int or goal_stall_limit < 1:
            raise ValueError("state goals require a positive per-goal stall limit")
        if goal_kind in ("trajectory", "hybrid") and commitment_steps != 1:
            raise ValueError("trajectory tracking replans after every command")
        wall_limit = 3600
    if not np.isfinite(max_seconds) or not 0 < max_seconds <= wall_limit:
        raise ValueError(f"hard wall budget must lie in (0,{wall_limit}]seconds")
    validate_attempt_budget(attempt_max_seconds)
    if not np.isfinite(control_timeout) or control_timeout <= 0:
        raise ValueError("control timeout must be positive and finite")
    if type(stride) is not int or stride < 1 or type(dwell) is not int or dwell < 1:
        raise ValueError("waypoint stride and dwell must be positive integers")
    lower = (0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0)
    upper = (0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0)
    config = WaypointConfig(
        horizon=horizon,
        candidates=candidates,
        iterations=iterations,
        max_steps=max_steps,
        commitment_steps=commitment_steps,
        lower_bounds=lower,
        upper_bounds=upper,
        goal_stall_limit=goal_stall_limit,
    )
    controller = asdict(config)
    if controller["goal_stall_limit"] is None:
        del controller["goal_stall_limit"]  # Keeps historical image plans byte-identical.
    attempts = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        reset = {
            "seed": seed,
            "object_xy": (np.array([0.34, -0.18]) + rng.uniform(-0.006, 0.006, 2)).tolist(),
            "plate_xy": (np.array([0.49, -0.09]) + rng.uniform(-0.006, 0.006, 2)).tolist(),
        }
        for mode in modes:
            attempts.append(
                {"attempt_id": f"{seed}-{mode}", "seed": seed, "mode": mode, "reset": reset}
            )
    plan = {
        "format_version": 1,
        "stage": stage,
        "seeds": seeds,
        "modes": modes,
        "max_seconds": max_seconds,
        "attempt_max_seconds": attempt_max_seconds,
        "finalization_reserve_seconds": min(5.0, max_seconds / 10),
        "control_timeout_seconds": float(control_timeout),
        "controller": controller,
        "waypoint_stride": stride,
        "waypoint_dwell": dwell,
        "demonstration_proposals": bool(proposals),
        "demo_selection": "lexicographically first successful NOMINAL apple/plate TRAIN episode",
        "threshold_rule": (
            "max(1.1*max(distance(frame[t+-2],frame[t])),"
            "median positive adjacent TRAIN-demo frame distance,1e-12,"
            "1.1*q90(successful NOMINAL TRAIN same-frame distances))"
        ),
        "proposal_rule": (
            "horizon-length applied-action windows entirely within prior stride frames; "
            "evenly spaced starts if budget limited"
        ),
        "reset_jitter_m": 0.006,
        "attempts": attempts,
        "success_rule": (
            "unchanged ordered AppleToPlateTask; scorer never drives waypoint advancement"
        ),
    }
    if goal_kind == "state":
        from embodied_jepa.models.state_goal_sensor import FIELDS, VISUAL_WEIGHT

        plan.update(
            goal_kind="state",
            backend=STATE_BACKEND,
            state_goal_fields=list(FIELDS),
            visual_diagnostic_weight=VISUAL_WEIGHT,
            goal_stall_limit=goal_stall_limit,
            demo_selection=(
                "per reset: successful NOMINAL apple/plate TRAIN episode whose frame-0 RGB has "
                "the smallest frozen sensor image distance to the live initial RGB; ties by "
                "lexicographic episode id; no state, object pose or score used for retrieval"
            ),
            goal_rule=(
                "measured right-arm+right-hand joint positions of the retrieved demo at frames "
                "stride, 2*stride, ... plus the final frame; frame 0 excluded"
            ),
            threshold_rule=(
                "state metric: max(1.1*max(distance(state[t+-2],state[t])),"
                "median positive adjacent TRAIN-demo state distance,1e-12,"
                "1.1*q90(successful NOMINAL TRAIN same-frame state distances))"
            ),
            proposal_rule="none: demonstration action proposals are disabled",
            stall_rule=(
                "terminate with goal_stall after goal_stall_limit acknowledged commands on one "
                "goal without advancing; recorded failure, no goal skipping"
            ),
            demo_replay_rule=(
                "NON-LEARNED control: replay the retrieved demo's recorded actions open-loop "
                "through the same bounds check and mandatory projection; no world model"
            ),
        )
    if goal_kind in ("trajectory", "hybrid"):
        from embodied_jepa.models.state_goal_sensor import FIELDS, VISUAL_WEIGHT

        plan.update(
            goal_kind=goal_kind,
            backend=STATE_BACKEND,
            state_goal_fields=list(FIELDS),
            visual_diagnostic_weight=VISUAL_WEIGHT,
            goal_stall_limit=goal_stall_limit,
            tracking_window=horizon,
            tracking_stall_commands=goal_stall_limit,
            tracking_stall_min_advance=stride,
            keyframe_threshold_fraction=KEYFRAME_FRACTION,
            demo_selection=(
                "per reset: successful NOMINAL apple/plate TRAIN episode whose frame-0 RGB has "
                "the smallest frozen sensor image distance to the live initial RGB; ties by "
                "lexicographic episode id; no state, object pose or score used for retrieval"
            ),
            reference_rule=(
                "measured right-arm+right-hand joint positions of the retrieved demo at every "
                "frame, keeping frame 0, the final frame and each frame whose state distance to "
                "the last kept frame exceeds keyframe_threshold_fraction * the demo's median "
                "positive adjacent-frame state distance (dead time removed)"
            ),
            cost_rule=(
                "mean over h=0..H-1 of the model state-goal distance between the state after "
                "step h and reference row min(t+1+h, L-1), t the measured reference index"
            ),
            progress_rule=(
                "t <- the latest index in [t, t+tracking_window] minimizing the measured "
                "state distance to the reference row (monotone, never decreases); "
                "reference_complete when t reaches the final row"
            ),
            stall_rule=(
                "terminate with reference_stall when the measured reference index gained fewer "
                "than tracking_stall_min_advance rows over the last tracking_stall_commands "
                "acknowledged commands; recorded failure"
            ),
            proposal_rule="none: no demonstration action proposals or action seeding",
            threshold_rule="none: no goal tolerances; progress is the nearest reference row",
        )
    if goal_kind == "hybrid":
        from embodied_jepa.hybrid_phase import HYBRID_LABEL
        from embodied_jepa.privileged_rollout import (
            PLANNING_DYNAMICS,
            PRIVILEGED_LABEL,
            REJECTED_ROLLOUT_COST,
        )

        # Mode-major order: all primary-arm attempts run first, so the secondary arm can
        # never consume the primary arm's budget.
        plan["attempts"] = sorted(attempts, key=lambda a: modes.index(a["mode"]))
        plan.update(
            privileged_ceiling=any(m in HYBRID_PRIVILEGED_MODES for m in modes),
            result_label=f"{HYBRID_LABEL}; {PRIVILEGED_LABEL}",
            planning_dynamics=PLANNING_DYNAMICS,
            rejected_rollout_cost=REJECTED_ROLLOUT_COST,
            attempt_order="mode-major in the declared mode order",
            hybrid_handoff_rows_before_close={
                "privileged_hybrid": 0,
                "privileged_hybrid_early": horizon,
            },
            handoff_rule=(
                "handoff_row = max(0, first_close_reference_index - rows_before_close); at "
                "the first observation whose measured reference index (the tracker's own "
                "progress rule) is >= handoff_row, planning stops for the rest of the attempt"
            ),
            replay_rule=(
                "after the handoff, replay the retrieved TRAIN demonstration's recorded "
                "actions open loop from action frames[matched row] to its last action, "
                "through the same bounds check and mandatory projection as demo_replay; "
                "ends with demo_exhausted; never a model result"
            ),
            privileged_rule=(
                "NON-LEARNED diagnostic ceiling: before the handoff each CEM candidate is "
                "scored by exact MuJoCo twin rollouts under the TASK-045 trajectory-tracking "
                "cost_rule; after it, recorded demonstration actions run open loop; never a "
                "learned result"
            ),
            demo_replay_rule=(
                "NON-LEARNED control: replay the retrieved demo's recorded actions open-loop "
                "from reset through the same bounds check and mandatory projection; no world "
                "model"
            ),
        )
    if privileged:
        from embodied_jepa.privileged_rollout import (
            PLANNING_DYNAMICS,
            PRIVILEGED_LABEL,
            REJECTED_ROLLOUT_COST,
        )

        plan.update(
            privileged_ceiling=True,
            result_label=PRIVILEGED_LABEL,
            planning_dynamics=PLANNING_DYNAMICS,
            rejected_rollout_cost=REJECTED_ROLLOUT_COST,
            privileged_rule=(
                "NON-LEARNED diagnostic ceiling: each CEM candidate is scored by restoring a "
                "copy of the live MuJoCo state into a non-rendering twin, executing its "
                "projected actions through the unchanged embodiment, and applying the same "
                "state-goal observed_distance at the horizon endpoint; never a learned result"
            )
            if goal_kind == "state"
            else (
                "NON-LEARNED diagnostic ceiling: each CEM candidate is scored by restoring a "
                "copy of the live MuJoCo state into a non-rendering twin, executing its "
                "projected actions through the unchanged embodiment, and applying the "
                "trajectory-tracking cost_rule to every reached state; never a learned result"
            ),
        )
    return plan


def wide_reset(seed):
    """TASK-047 wide-jitter reset: apple +-3 cm, plate +-2 cm around the unchanged centres."""
    rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "object_xy": (
            np.array(RESET_CENTERS["object_xy"])
            + rng.uniform(-WIDE_JITTER_M["object_xy"], WIDE_JITTER_M["object_xy"], 2)
        ).tolist(),
        "plate_xy": (
            np.array(RESET_CENTERS["plate_xy"])
            + rng.uniform(-WIDE_JITTER_M["plate_xy"], WIDE_JITTER_M["plate_xy"], 2)
        ).tolist(),
    }


def make_object_plan(
    *,
    stage,
    seeds,
    modes,
    max_seconds,
    max_steps,
    stride,
    dwell,
    horizon,
    candidates,
    iterations,
    proposals,
    control_timeout,
    commitment_steps,
    attempt_max_seconds,
    goal_stall_limit,
):
    """TASK-047 plan: NON-LEARNED arms on the wide-jitter development resets only."""
    from embodied_jepa.object_ceiling import (
        OBJECT_CEILING_LABEL,
        PHASES,
        PLANNING_DYNAMICS,
        ObjectCeilingConfig,
    )
    from embodied_jepa.privileged_rollout import REJECTED_ROLLOUT_COST

    if stage != "development":
        raise ValueError("the wide-jitter distribution is a development distribution only")
    seeds = list(WIDE_COHORT if seeds is None else seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s not in WIDE_COHORT for s in seeds)
    ):
        raise ValueError("seeds must be unique members of the wide-jitter development cohort")
    modes = list(OBJECT_MODES if modes is None else modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in OBJECT_MODES for m in modes):
        raise ValueError("object plans run demo_replay/scripted_oracle/privileged_object only")
    if proposals:
        raise ValueError("the object-aware protocol forbids demonstration action proposals")
    if commitment_steps != 1:
        raise ValueError("the object-aware ceiling replans after every command")
    if goal_stall_limit is not None:
        raise ValueError("goal stall limits belong to the state-goal protocols")
    if not np.isfinite(max_seconds) or not 0 < max_seconds <= OBJECT_WALL_LIMIT:
        raise ValueError(f"hard wall budget must lie in (0,{OBJECT_WALL_LIMIT}]seconds")
    validate_attempt_budget(attempt_max_seconds)
    if not np.isfinite(control_timeout) or control_timeout <= 0:
        raise ValueError("control timeout must be positive and finite")
    if type(stride) is not int or stride < 1 or type(dwell) is not int or dwell < 1:
        raise ValueError("library stride and dwell must be positive integers")
    lower = (0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0)
    upper = (0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0)
    controller = asdict(
        WaypointConfig(
            horizon=horizon,
            candidates=candidates,
            iterations=iterations,
            max_steps=max_steps,
            commitment_steps=1,
            lower_bounds=lower,
            upper_bounds=upper,
        )
    )
    ceiling = asdict(
        ObjectCeilingConfig(
            horizon=horizon, candidates=candidates, iterations=iterations, max_steps=max_steps
        )
    )
    del ceiling["seed"]  # Each attempt seeds its ceiling with its own reset seed.
    attempts = [
        {"attempt_id": f"{seed}-{mode}", "seed": seed, "mode": mode, "reset": wide_reset(seed)}
        for mode in modes
        for seed in seeds
    ]
    return {
        "format_version": 1,
        "stage": stage,
        "distribution": "apple_plate_wide_jitter_dev_v1",
        "goal_kind": "object",
        "seeds": seeds,
        "modes": modes,
        "max_seconds": max_seconds,
        "attempt_max_seconds": attempt_max_seconds,
        "finalization_reserve_seconds": min(5.0, max_seconds / 10),
        "control_timeout_seconds": float(control_timeout),
        "controller": controller,
        "object_ceiling": ceiling,
        "object_ceiling_phases": list(PHASES),
        "waypoint_stride": stride,
        "waypoint_dwell": dwell,
        "demonstration_proposals": False,
        "backend": STATE_BACKEND,
        "reset_centers_xy": {k: list(v) for k, v in RESET_CENTERS.items()},
        "reset_jitter_m": dict(WIDE_JITTER_M),
        "reset_rule": (
            "rng=numpy.random.default_rng(seed); object_xy=centre+rng.uniform(-0.03,0.03,2); "
            "plate_xy=centre+rng.uniform(-0.02,0.02,2)"
        ),
        "attempts": attempts,
        "attempt_order": "mode-major in the declared mode order",
        "privileged_ceiling": OBJECT_MODE in modes,
        "result_label": OBJECT_CEILING_LABEL,
        "planning_dynamics": PLANNING_DYNAMICS,
        "rejected_rollout_cost": REJECTED_ROLLOUT_COST,
        "success_rule": "unchanged ordered AppleToPlateTask; grasp stage = reach, >=5 cm lift "
        "and hand contact",
        "demo_selection": (
            "per reset: successful NOMINAL apple/plate TRAIN episode whose frame-0 RGB has "
            "the smallest frozen sensor image distance to the live initial RGB; ties by "
            "lexicographic episode id; no state, object pose or score used for retrieval"
        ),
        "demo_replay_rule": (
            "NON-LEARNED control: replay the retrieved demo's recorded actions open-loop "
            "from reset through the same bounds check and mandatory projection; no world model"
        ),
        "scripted_rule": (
            "NON-LEARNED privileged reference: the TRAIN collector's scripted policy "
            "(scripted.apple_collector_policy, initial simulator truth) through the same bounds "
            "check and mandatory projection; ends with policy_complete; feasibility reference"
        ),
        "privileged_rule": (
            "NON-LEARNED diagnostic ceiling: per command, CEM candidates are projected, "
            "executed in an exact non-rendering MuJoCo twin, and scored by the mean over the "
            "horizon of an object-aware phase cost on simulator apple/palm/plate state; phase "
            "transitions read live simulator truth; never a learned result"
        ),
        "gate_rule": (
            "privileged_object grasp resets >= 6 of 8 AND demo_replay grasp resets <= "
            "privileged_object grasp resets - 3, with zero robot and full-state rollout parity "
            "mismatches over non-vacuous checks (>= executed commands - 1 per attempt), all "
            "ceiling and replay attempts counted and valid provenance"
        ),
    }


def checkpoint_model(dataset, checkpoint, *, device="cpu", backend=None):
    import torch

    from embodied_jepa.config import MODELS
    from embodied_jepa.data import DatasetStore

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    store = DatasetStore(dataset)
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "dataset_hash": store.manifest_hash,
        "split_hash": json_hash(
            {"splits": store.manifest.get("splits"), "policy": store.manifest.get("split_policy")}
        ),
        "action_hash": json_hash(store.manifest["action_manifest"]),
    }
    for key, value in expected.items():
        if envelope.get("metadata", {}).get(key) != value:
            raise ContractError(f"checkpoint {key} does not match sealed dataset")
    if json_hash(envelope.get("state_schema")) != json_hash(asdict(store.state_schema)):
        raise ContractError("checkpoint robot-state schema mismatch")
    training = set(store.manifest.get("splits", {}).get("train", []))
    normalized = set(envelope.get("metadata", {}).get("normalization", {}).get("episode_ids", []))
    if not training or normalized != training:
        raise ContractError("checkpoint normalization must cover exactly the frozen training split")
    if backend is None:
        model = MODELS.create(
            envelope["backend"],
            state_schema=store.state_schema,
            device=device,
            seed=envelope["seed"],
            config=envelope["config"],
            metadata=expected,
        )
    elif backend == STATE_BACKEND:
        # Frozen composition over the unchanged sensor checkpoint: no new weights.
        model = MODELS.create(
            backend,
            state_schema=store.state_schema,
            device=device,
            seed=envelope["seed"],
            config={"sensor_config": envelope["config"]},
            metadata=expected,
        )
    else:
        raise ContractError("unsupported evaluation backend override")
    model.load(checkpoint)
    if not callable(getattr(model, "observed_distance", None)):
        raise ContractError("waypoint model requires an observed-image distance metric")
    return store, model


def select_demonstration(store):
    for episode_id in sorted(store.manifest.get("splits", {}).get("train", [])):
        episode = store.read_episode(episode_id)
        if (episode.object_id, episode.container_id) != ("apple", "plate"):
            continue
        if (
            episode.metadata.get("collection_success") is True
            and episode.terminated
            and episode.metadata.get("variant") == "nominal"
        ):
            return episode
    raise ContractError("no successful apple/plate training demonstration")


def nominal_calibration_episodes(store):
    result = []
    for name in sorted(store.manifest.get("splits", {}).get("train", [])):
        episode = store.read_episode(name)
        if (
            (episode.object_id, episode.container_id) == ("apple", "plate")
            and episode.metadata.get("variant") == "nominal"
            and episode.metadata.get("collection_success") is True
            and episode.terminated
        ):
            result.append(episode)
    return result


def calibrate_waypoints(
    model,
    episode,
    *,
    stride=10,
    dwell=3,
    horizon=4,
    candidates=16,
    proposals=True,
    calibration_episodes=None,
    training_episode_ids=None,
):
    references = list(calibration_episodes) if calibration_episodes is not None else [episode]
    allowed = (
        set(training_episode_ids) if training_episode_ids is not None else {episode.episode_id}
    )
    if episode.episode_id not in allowed or any(e.episode_id not in allowed for e in references):
        raise ContractError("waypoint calibration received non-training episode")
    frames = episode.observations["onboard_rgb"]
    indices = list(range(0, len(frames), stride))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    def distance(i, j):
        cost = np.asarray(
            model.observed_distance(
                {"onboard_rgb": frames[i : i + 1]}, {"onboard_rgb": frames[j : j + 1]}
            )
        )
        if cost.shape != (1,) or not np.isfinite(cost).all() or cost[0] < 0:
            raise ContractError("invalid training image calibration distance")
        return float(cost[0])

    adjacent = [distance(i - 1, i) for i in range(1, len(frames))]
    positive = [v for v in adjacent if v > 0]
    floor = max(float(np.median(positive)) if positive else 0.0, 1e-12)
    waypoints, records = [], []
    for waypoint_number, index in enumerate(indices):
        neighborhood = list(range(max(0, index - 2), min(len(frames), index + 3)))
        distances = [distance(i, index) for i in neighborhood]
        cross = []
        for reference in references:
            source = reference.observations["onboard_rgb"]
            if index >= len(source):
                continue  # No end clamping: only observed corresponding frames.
            value = np.asarray(
                model.observed_distance(
                    {"onboard_rgb": source[index : index + 1]},
                    {"onboard_rgb": frames[index : index + 1]},
                )
            )
            if value.shape != (1,) or not np.isfinite(value).all() or value[0] < 0:
                raise ContractError("invalid cross-training image distance")
            cross.append(
                {"episode_id": reference.episode_id, "frame": index, "distance": float(value[0])}
            )
        if not cross:
            raise ContractError("waypoint has no training frame coverage")
        cross_quantile = float(np.quantile([v["distance"] for v in cross], 0.9))
        threshold = max(1.1 * max(distances), floor, 1.1 * cross_quantile)
        starts = list(range(max(0, index - stride), index - horizon + 1)) if proposals else []
        if len(starts) > candidates - 2:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, candidates - 2, dtype=int)]
        action_proposals = (
            np.stack([episode.actions[i : i + horizon] for i in starts]) if starts else None
        )
        waypoint = ImageWaypoint(
            {"onboard_rgb": frames[index : index + 1]},
            threshold,
            dwell,
            f"{episode.episode_id}/frame-{index}",
            action_proposals=action_proposals,
        )
        waypoints.append(waypoint)
        neighbors = [i for i in (waypoint_number - 1, waypoint_number + 1) if 0 <= i < len(indices)]
        separability = [
            {"frame": indices[i], "distance": distance(indices[i], index)} for i in neighbors
        ]
        for entry in separability:
            entry["inside_threshold"] = entry["distance"] <= threshold
        records.append(
            {
                "frame": index,
                "adjacent_waypoint_separability": separability,
                "overlapping_adjacent_waypoint_count": sum(
                    e["inside_threshold"] for e in separability
                ),
                "threshold": threshold,
                "neighborhood_frames": neighborhood,
                "neighborhood_distances": distances,
                "cross_training_distances": cross,
                "cross_training_q90": cross_quantile,
                "cross_training_reference_count": len(cross),
                "proposal_start_frames": starts,
                "image_sha256": waypoint.image_hash,
                "source_id": waypoint.source_id,
            }
        )
    return waypoints, {
        "episode_id": episode.episode_id,
        "adjacent_distance_floor": floor,
        "adjacent_distances": adjacent,
        "calibration_training_episode_ids": [e.episode_id for e in references],
        "waypoints_with_adjacent_overlap": sum(
            r["overlapping_adjacent_waypoint_count"] > 0 for r in records
        ),
        "waypoints": records,
    }


def _state_rows(episode, rows):
    from embodied_jepa.contracts import RobotState

    rows = np.asarray(rows, dtype=int)
    return RobotState(
        np.asarray(episode.robot_states)[rows],
        np.asarray(episode.state_mask)[rows],
        np.asarray(episode.timestamps, dtype=np.float64)[rows],
        episode.state_schema,
    )


def _state_distance(model, current_episode, current_rows, goal_values):
    cost = np.asarray(
        model.observed_distance(_state_rows(current_episode, current_rows), goal_values)
    )
    if cost.shape != (len(current_rows),) or not np.isfinite(cost).all() or (cost < 0).any():
        raise ContractError("invalid training state calibration distance")
    return [float(v) for v in cost]


def state_goal_frames(length, stride):
    frames = list(range(stride, length, stride))
    if not frames or frames[-1] != length - 1:
        frames.append(length - 1)
    return frames


def build_state_goal_library(model, demonstrations, *, stride, dwell, training_episode_ids):
    """TRAIN-only state goals and tolerances for every candidate demonstration.

    Each candidate is a successful nominal TRAIN demonstration; the same set is the
    cross-demonstration reference population of the existing threshold recipe.
    """
    demonstrations = sorted(demonstrations, key=lambda e: e.episode_id)
    allowed = set(training_episode_ids)
    if not demonstrations or any(e.episode_id not in allowed for e in demonstrations):
        raise ContractError("state-goal library received a non-training episode")
    if len({e.episode_id for e in demonstrations}) != len(demonstrations):
        raise ContractError("duplicate state-goal demonstration")
    if type(stride) is not int or stride < 1 or type(dwell) is not int or dwell < 1:
        raise ContractError("state-goal stride and dwell must be positive integers")
    arrays, records = {}, []
    for number, episode in enumerate(demonstrations):
        states = np.asarray(episode.robot_states)
        length = len(states)
        frames = episode.observations["onboard_rgb"]
        if len(frames) != length or len(episode.actions) != length - 1 or length < 2:
            raise ContractError("demonstration needs T actions and T+1 synchronized frames")
        values = model.select(states, episode.state_schema)
        adjacent = _state_distance(model, episode, range(1, length), values[:-1])
        positive = [v for v in adjacent if v > 0]
        floor = max(float(np.median(positive)) if positive else 0.0, 1e-12)
        goal_frames = state_goal_frames(length, stride)
        goals = []
        for goal_number, index in enumerate(goal_frames):
            neighborhood = list(range(max(0, index - 2), min(length, index + 3)))
            target = values[index : index + 1]
            distances = _state_distance(
                model, episode, neighborhood, np.repeat(target, len(neighborhood), 0)
            )
            cross = []
            for reference in demonstrations:
                if index >= len(reference.robot_states):
                    continue  # No end clamping: only observed corresponding frames.
                cross.append(
                    {
                        "episode_id": reference.episode_id,
                        "frame": index,
                        "distance": _state_distance(model, reference, [index], target)[0],
                    }
                )
            cross_quantile = float(np.quantile([v["distance"] for v in cross], 0.9))
            threshold = max(1.1 * max(distances), floor, 1.1 * cross_quantile)
            previous = goal_frames[goal_number - 1] if goal_number else 0
            neighbors = [previous] + (
                [goal_frames[goal_number + 1]] if goal_number + 1 < len(goal_frames) else []
            )
            separability = [
                {
                    "frame": frame,
                    "distance": (d := _state_distance(model, episode, [frame], target)[0]),
                    "inside_threshold": d <= threshold,
                }
                for frame in neighbors
            ]
            goals.append(
                {
                    "frame": index,
                    "threshold": threshold,
                    "dwell": dwell,
                    "source_id": f"{episode.episode_id}/frame-{index}",
                    "neighborhood_frames": neighborhood,
                    "neighborhood_distances": distances,
                    "cross_training_distances": cross,
                    "cross_training_q90": cross_quantile,
                    "adjacent_goal_separability": separability,
                    "overlapping_adjacent_goal_count": sum(
                        e["inside_threshold"] for e in separability
                    ),
                    "state_sha256": hashlib.sha256(target.astype("<f4").tobytes()).hexdigest(),
                }
            )
        actions = np.asarray(episode.actions, dtype=np.float32)
        right_grasp = EE_DELTA_GRASP_V0.names.index("right_grasp")
        # A "close goal" is preceded by a recorded right-grasp command above -1 in the
        # stride window before it; derived from the TRAIN demonstration only.
        close_goals = [
            goal_number
            for goal_number, index in enumerate(goal_frames)
            if (actions[max(0, index - stride) : index, right_grasp] > -1.0).any()
        ]
        arrays[f"initial_{number}"] = frames[0:1]
        arrays[f"goals_{number}"] = values[goal_frames]
        arrays[f"goal_images_{number}"] = frames[goal_frames]
        arrays[f"actions_{number}"] = actions
        records.append(
            {
                "index": number,
                "episode_id": episode.episode_id,
                "split": "train",
                "variant": episode.metadata.get("variant"),
                "collection_success": episode.metadata.get("collection_success"),
                "length": length,
                "goal_frames": goal_frames,
                "adjacent_distance_floor": floor,
                "first_close_goal_index": close_goals[0] if close_goals else None,
                "goals_with_adjacent_overlap": sum(
                    g["overlapping_adjacent_goal_count"] > 0 for g in goals
                ),
                "goals": goals,
            }
        )
    return arrays, {
        "format_version": 1,
        "stride": stride,
        "dwell": dwell,
        "training_episode_ids_sha256": json_hash(sorted(allowed)),
        "calibration_training_episode_ids": [e.episode_id for e in demonstrations],
        "demonstrations": records,
    }


def build_tracking_references(model, demonstrations, library, *, fraction, training_episode_ids):
    """TRAIN-only per-frame references for every library demonstration (TASK-045).

    Rows are the measured right-arm+hand positions at the kept frames of
    ``keyframe_indices`` with threshold ``fraction * adjacent_distance_floor``.
    The recorded TRAIN scorer labels give ``demo_grasp_reference_index`` for a
    post-hoc report reading only; they never enter planning or progress.
    """
    from embodied_jepa.trajectory_tracking import TrackingReference, keyframe_indices

    by_id = {e.episode_id: e for e in demonstrations}
    allowed = set(training_episode_ids)
    if not np.isfinite(fraction) or fraction < 0:
        raise ContractError("keyframe fraction must be finite and nonnegative")
    arrays, records = {}, []
    right_grasp = EE_DELTA_GRASP_V0.names.index("right_grasp")
    for record in library["demonstrations"]:
        episode = by_id.get(record["episode_id"])
        if episode is None or episode.episode_id not in allowed:
            raise ContractError("tracking reference received a non-training episode")
        values = model.select(np.asarray(episode.robot_states), episode.state_schema)
        threshold = fraction * record["adjacent_distance_floor"]
        frames = keyframe_indices(
            len(values),
            lambda i, j, e=episode, v=values: _state_distance(model, e, [i], v[j : j + 1])[0],
            threshold,
        )
        actions = np.asarray(episode.actions, dtype=np.float32)
        closing = np.flatnonzero(actions[:, right_grasp] > -1.0)
        # Action i produces frame i + 1; rows at or after the first closing frame.
        close_frame = int(closing[0]) + 1 if len(closing) else None
        scores = episode.metadata.get("stage_scores") or []
        grasps = [i for i, row in enumerate(scores) if isinstance(row, dict) and row.get("grasp")]
        grasp_frame = grasps[0] + 1 if grasps else None

        def first_row(frame, frames=frames):
            return None if frame is None else next(n for n, f in enumerate(frames) if f >= frame)

        arrays[f"reference_{record['index']}"] = values[frames]
        arrays[f"frames_{record['index']}"] = np.asarray(frames, np.int64)
        records.append(
            {
                "index": record["index"],
                "episode_id": episode.episode_id,
                "length": len(values),
                "keyframe_threshold": threshold,
                "reference_length": len(frames),
                "dropped_frames": len(values) - len(frames),
                "first_close_frame": close_frame,
                "first_close_reference_index": first_row(close_frame),
                "demo_grasp_frame": grasp_frame,
                "demo_grasp_reference_index": first_row(grasp_frame),
                # Same definition as TrackingReference.sha256 in the controller trace.
                "reference_sha256": TrackingReference(
                    values[frames], tuple(frames), f"{episode.episode_id}/keyframes"
                ).sha256,
                "final_row_hold": frames[-1] - frames[-2],
            }
        )
    return arrays, {
        "format_version": 1,
        "keyframe_threshold_fraction": fraction,
        "training_episode_ids_sha256": json_hash(sorted(allowed)),
        "references": records,
    }


def load_tracking_references(output, plan):
    for name, field in (
        ("tracking_references.npz", "tracking_references_sha256"),
        ("tracking_calibration.json", "tracking_calibration_sha256"),
    ):
        if digest(Path(output) / name) != plan[field]:
            raise ContractError("frozen tracking reference changed")
    calibration = json.loads((Path(output) / "tracking_calibration.json").read_text())
    with np.load(Path(output) / "tracking_references.npz", allow_pickle=False) as arrays:
        calibration["arrays"] = {key: arrays[key] for key in arrays.files}
    return calibration


def tracking_reference(references, record):
    from embodied_jepa.trajectory_tracking import TrackingReference

    row = next(r for r in references["references"] if r["index"] == record["index"])
    if row["episode_id"] != record["episode_id"]:
        raise ContractError("tracking reference and retrieved demonstration differ")
    arrays = references["arrays"]
    return TrackingReference(
        arrays[f"reference_{row['index']}"],
        tuple(int(f) for f in arrays[f"frames_{row['index']}"]),
        f"{row['episode_id']}/keyframes",
    ), row


def retrieve_demonstration(model, images, library):
    """Nearest candidate by initial RGB only; ties resolve to the lexical first id."""
    distances = []
    for record in library["demonstrations"]:
        initial = library["arrays"][f"initial_{record['index']}"]
        value = np.asarray(
            model.image_distance({"onboard_rgb": images["onboard_rgb"]}, {"onboard_rgb": initial})
        )
        if value.shape != (1,) or not np.isfinite(value).all() or value[0] < 0:
            raise ContractError("invalid retrieval image distance")
        distances.append(float(value[0]))
    chosen = int(np.argmin(distances))
    record = library["demonstrations"][chosen]
    return record, {
        "rule": "min frozen sensor image distance to demonstration frame 0; lexical tie-break",
        "demonstration_episode_id": record["episode_id"],
        "distances": {
            r["episode_id"]: d for r, d in zip(library["demonstrations"], distances, strict=True)
        },
        "input_image_sha256": hashlib.sha256(images["onboard_rgb"].tobytes()).hexdigest(),
    }


def state_waypoints(library, record):
    arrays = library["arrays"]
    goals, images = arrays[f"goals_{record['index']}"], arrays[f"goal_images_{record['index']}"]
    if len(goals) != len(record["goals"]):
        raise ContractError("state-goal array/calibration count differs")
    return [
        StateWaypoint(
            goals[i : i + 1],
            row["threshold"],
            row["dwell"],
            row["source_id"],
            images={"onboard_rgb": images[i : i + 1]},
        )
        for i, row in enumerate(record["goals"])
    ]


def load_state_library(output, plan):
    for name, field in (
        ("state_goals.npz", "state_goals_sha256"),
        ("state_calibration.json", "state_calibration_sha256"),
    ):
        if digest(Path(output) / name) != plan[field]:
            raise ContractError("frozen state-goal artifact changed")
    calibration = json.loads((Path(output) / "state_calibration.json").read_text())
    with np.load(Path(output) / "state_goals.npz", allow_pickle=False) as arrays:
        calibration["arrays"] = {key: arrays[key] for key in arrays.files}
    return calibration


def ordered_stage_count(score):
    count = 0
    for stage in STAGES:
        if not (score or {}).get(stage):
            break
        count += 1
    return count


FAILED_TERMINATIONS = ("runtime_error", "deadline_miss", "attempt_timeout")


def counted_attempt(record):
    """Only cleanly completed, provenance-valid attempts contribute scored stages."""
    return (
        record is not None
        and record.get("status", "completed") == "completed"
        and record.get("termination_reason") not in FAILED_TERMINATIONS
        and record.get("provenance_valid", True)
    )


def state_gate(records, seeds):
    """Preregistered TASK-043 gate; missing/failed attempts count as zero stages."""
    by = {(r["seed"], r["mode"]): r for r in records}

    def score(seed, mode):
        record = by.get((seed, mode))
        return (record.get("score") or {}) if counted_attempt(record) else {}

    def stages(mode):
        return sum(ordered_stage_count(score(s, mode)) for s in seeds)

    def resets(mode, stage):
        return sum(bool(score(s, mode).get(stage)) for s in seeds)

    grasps = resets("learned", "grasp")
    sums = {mode: stages(mode) for mode in STATE_MODES}
    provenance_valid = all(r.get("provenance_valid", True) for r in records)
    primary = (
        provenance_valid
        and grasps >= 2
        and sums["learned"] > sums["dynamics_shuffle"]
        and sums["learned"] > sums["persistence"]
    )

    def stalled_before_close(seed):
        record = by.get((seed, "learned")) or {}
        close = record.get("first_close_goal_index")
        reached = record.get("last_goal_index")
        return not score(seed, "learned").get("grasp") and (
            close is None or reached is None or reached < close
        )

    stalled = sum(stalled_before_close(s) for s in seeds)
    learned_reach = resets("learned", "reach")
    return {
        "learned_grasp_resets": grasps,
        "summed_ordered_stages": sums,
        "counted_attempts": sum(counted_attempt(r) for r in records),
        "provenance_valid": provenance_valid,
        "primary_gate_passed": bool(primary),
        "learned_full_successes": sum(
            bool(by.get((s, "learned"), {}).get("success")) and counted_attempt(by[(s, "learned")])
            for s in seeds
        ),
        "falsification": {
            "learned_stalled_before_close_goal_resets": stalled,
            "model_or_planner_failure": stalled >= 3,
            "learned_reach_resets": learned_reach,
            "grasp_precision_bottleneck": learned_reach > 0 and grasps == 0,
            # "~=" is read as "not strictly above"; applies only when learned scored stages.
            "scaffold_explains_result": sums["learned"] > 0
            and (
                sums["learned"] <= sums["dynamics_shuffle"]
                or (
                    sums["demo_replay"] >= sums["learned"]
                    and sums["learned"] <= max(sums["dynamics_shuffle"], sums["persistence"])
                )
            ),
        },
        "rule": (
            "learned grasp on >=2 resets AND learned summed ordered stages > dynamics_shuffle "
            "and > persistence; only cleanly completed provenance-valid attempts are scored; "
            "demo_replay is a non-learned reference only"
        ),
    }


def ceiling_gate(records, seeds):
    """Preregistered TASK-044 ceiling gate; missing/failed attempts count as zero stages."""
    by = {(r["seed"], r["mode"]): r for r in records if r.get("mode") == PRIVILEGED_MODE}
    per_reset = {}
    for seed in seeds:
        record = by.get((seed, PRIVILEGED_MODE))
        counted = counted_attempt(record)
        score = (record.get("score") or {}) if counted else {}
        record = record or {}
        close, reached = record.get("first_close_goal_index"), record.get("last_goal_index")
        per_reset[str(seed)] = {
            "counted": counted,
            "status": record.get("status"),
            "termination_reason": record.get("termination_reason"),
            "executed_steps": record.get("executed_steps"),
            "demonstration_episode_id": record.get("demonstration_episode_id"),
            "goal_count": record.get("goal_count"),
            "last_goal_index": reached,
            "first_close_goal_index": close,
            "ordered_stages": ordered_stage_count(score),
            "grasp": bool(score.get("grasp")),
            "success": bool(score.get("success")),
            "reported_ordered_stages_uncounted": ordered_stage_count(record.get("score")),
            "rollout_parity_checks": record.get("rollout_parity_checks"),
            "rollout_parity_mismatches": record.get("rollout_parity_mismatches"),
            # Readings use counted attempts only; last_goal_index is the goal being pursued.
            "stalled_before_close_goal": counted
            and not score.get("grasp")
            and (close is None or reached is None or reached < close),
            "advanced_past_close_goal_without_grasp": counted
            and not score.get("grasp")
            and close is not None
            and reached is not None
            and reached > close,
        }
    rows = list(per_reset.values())
    grasps = sum(r["grasp"] for r in rows)
    counted_attempts = sum(r["counted"] for r in rows)
    provenance_valid = all(r.get("provenance_valid", True) for r in records)
    mismatches = sum(r["rollout_parity_mismatches"] or 0 for r in rows if r["counted"])
    exact = mismatches == 0 and all(
        r["rollout_parity_mismatches"] is not None for r in rows if r["counted"]
    )
    passed = provenance_valid and exact and grasps >= 2
    conclusive = provenance_valid and exact and counted_attempts == len(seeds)
    stalled = sum(r["stalled_before_close_goal"] for r in rows)
    past_close = sum(r["advanced_past_close_goal_without_grasp"] for r in rows)
    return {
        "label": "NON-LEARNED privileged MuJoCo-rollout planning ceiling; not a learned result",
        "privileged_grasp_resets": grasps,
        "summed_ordered_stages": sum(r["ordered_stages"] for r in rows),
        "full_successes": sum(r["success"] for r in rows),
        "counted_attempts": counted_attempts,
        "provenance_valid": provenance_valid,
        "rollout_parity_mismatches": mismatches,
        "rollouts_exact": bool(exact),
        "primary_gate_passed": bool(passed),
        "readings": {
            "conclusive": bool(passed or conclusive),
            "stalled_before_close_goal_resets": stalled,
            "state_goal_tracking_inadequate": conclusive and not passed and stalled >= 3,
            "advanced_past_close_goal_without_grasp_resets": past_close,
            "arm_pose_goals_insufficient_for_grasp": conclusive and not passed and past_close >= 3,
            "interpretation": (
                "goal/planner design adequate under perfect dynamics; learned dynamics are "
                "the bottleneck"
                if passed
                else "planner/goal design must change before further model training"
                if conclusive
                else "inconclusive: missing/failed attempts, invalid provenance or inexact rollouts"
            ),
        },
        "per_reset": per_reset,
        "rule": (
            "privileged_rollout reaches the unchanged scorer's grasp stage on >=2 resets with "
            "zero rollout parity mismatches; only cleanly completed provenance-valid attempts "
            "are scored; readings are conclusive only with all attempts counted; non-learned "
            "ceiling only"
        ),
    }


def _tracking_row(record, counted):
    score = (record.get("score") or {}) if counted else {}
    record = record or {}
    close = record.get("first_close_reference_index")
    grasp_row = record.get("demo_grasp_reference_index")
    reached = record.get("last_reference_index")
    return {
        "counted": counted,
        "status": record.get("status"),
        "termination_reason": record.get("termination_reason"),
        "executed_steps": record.get("executed_steps"),
        "demonstration_episode_id": record.get("demonstration_episode_id"),
        "reference_length": record.get("reference_length"),
        "last_reference_index": reached,
        "first_close_reference_index": close,
        "demo_grasp_reference_index": grasp_row,
        "ordered_stages": ordered_stage_count(score),
        "grasp": bool(score.get("grasp")),
        "success": bool(score.get("success")),
        "reported_ordered_stages_uncounted": ordered_stage_count(record.get("score")),
        # Readings use counted attempts only; the index is the measured reference row.
        "stalled_before_close_reference": counted
        and not score.get("grasp")
        and (close is None or reached is None or reached < close),
        "passed_demo_grasp_reference_without_grasp": counted
        and not score.get("grasp")
        and grasp_row is not None
        and reached is not None
        and reached >= grasp_row,
    }


def tracking_ceiling_gate(records, seeds):
    """Preregistered TASK-045 ceiling gate; missing/failed attempts count as zero stages."""
    by = {(r["seed"], r["mode"]): r for r in records if r.get("mode") == PRIVILEGED_MODE}
    per_reset = {}
    for seed in seeds:
        record = by.get((seed, PRIVILEGED_MODE))
        row = _tracking_row(record, counted_attempt(record))
        row.update(
            rollout_parity_checks=(record or {}).get("rollout_parity_checks"),
            rollout_parity_mismatches=(record or {}).get("rollout_parity_mismatches"),
        )
        per_reset[str(seed)] = row
    rows = list(per_reset.values())
    grasps = sum(r["grasp"] for r in rows)
    counted_attempts = sum(r["counted"] for r in rows)
    provenance_valid = all(r.get("provenance_valid", True) for r in records)
    mismatches = sum(r["rollout_parity_mismatches"] or 0 for r in rows if r["counted"])
    exact = mismatches == 0 and all(
        r["rollout_parity_mismatches"] is not None for r in rows if r["counted"]
    )
    passed = provenance_valid and exact and grasps >= 2
    conclusive = provenance_valid and exact and counted_attempts == len(seeds)
    stalled = sum(r["stalled_before_close_reference"] for r in rows)
    past = sum(r["passed_demo_grasp_reference_without_grasp"] for r in rows)
    return {
        "label": "NON-LEARNED privileged MuJoCo-rollout trajectory-tracking ceiling; "
        "not a learned result",
        "privileged_grasp_resets": grasps,
        "summed_ordered_stages": sum(r["ordered_stages"] for r in rows),
        "full_successes": sum(r["success"] for r in rows),
        "counted_attempts": counted_attempts,
        "provenance_valid": provenance_valid,
        "rollout_parity_mismatches": mismatches,
        "rollouts_exact": bool(exact),
        "primary_gate_passed": bool(passed),
        "learned_stage_authorized": bool(passed),
        "readings": {
            "conclusive": bool(conclusive),
            "stalled_before_close_reference_resets": stalled,
            "trajectory_tracking_inadequate": conclusive and not passed and stalled >= 3,
            "passed_demo_grasp_reference_without_grasp_resets": past,
            "arm_pose_tracking_insufficient_for_grasp": conclusive and not passed and past >= 3,
            "interpretation": (
                "trajectory-tracking scaffold adequate under perfect dynamics; run the "
                "preregistered learned stage"
                if passed and conclusive
                else "trajectory-tracking scaffold fails under perfect dynamics; do not pair it "
                "with learned dynamics"
                if conclusive
                else "inconclusive: missing/failed attempts, invalid provenance or inexact rollouts"
            ),
        },
        "per_reset": per_reset,
        "rule": (
            "privileged_rollout trajectory tracking reaches the unchanged scorer's grasp stage "
            "on >=2 resets with zero rollout parity mismatches; only cleanly completed "
            "provenance-valid attempts are scored; readings are conclusive only with all "
            "attempts counted; non-learned ceiling only"
        ),
    }


def tracking_gate(records, seeds):
    """Preregistered TASK-045 secondary learned stage; runs only after a ceiling pass."""
    by = {(r["seed"], r["mode"]): r for r in records}
    per_mode = {
        mode: {
            str(seed): _tracking_row(by.get((seed, mode)), counted_attempt(by.get((seed, mode))))
            for seed in seeds
        }
        for mode in TRACKING_MODES
    }
    sums = {m: sum(r["ordered_stages"] for r in rows.values()) for m, rows in per_mode.items()}
    grasps = sum(r["grasp"] for r in per_mode["learned"].values())
    provenance_valid = all(r.get("provenance_valid", True) for r in records)
    counted = sum(r["counted"] for rows in per_mode.values() for r in rows.values())
    passed = (
        provenance_valid
        and grasps >= 2
        and sums["learned"] > sums["dynamics_shuffle"]
        and sums["learned"] > sums["persistence"]
    )
    conclusive = provenance_valid and counted == len(seeds) * len(TRACKING_MODES)
    stalled = sum(r["stalled_before_close_reference"] for r in per_mode["learned"].values())
    return {
        "learned_grasp_resets": grasps,
        "summed_ordered_stages": sums,
        "counted_attempts": counted,
        "provenance_valid": provenance_valid,
        "primary_gate_passed": bool(passed),
        "learned_full_successes": sum(r["success"] for r in per_mode["learned"].values()),
        "readings": {
            "conclusive": bool(conclusive),
            "learned_stalled_before_close_reference_resets": stalled,
            "learned_dynamics_failure_under_tracking": conclusive and not passed and stalled >= 3,
            "no_model_contribution": conclusive
            and sums["learned"] > 0
            and sums["learned"] <= max(sums["dynamics_shuffle"], sums["persistence"]),
        },
        "per_mode": per_mode,
        "rule": (
            "learned grasp on >=2 resets AND learned summed ordered stages > dynamics_shuffle "
            "and > persistence; only cleanly completed provenance-valid attempts are scored"
        ),
    }


def _hybrid_arm(records, seeds, mode):
    """Per-reset rows for one hybrid-plan mode; counted attempts only drive readings."""
    by = {(r["seed"], r["mode"]): r for r in records if r.get("mode") == mode}
    per_reset = {}
    for seed in seeds:
        record = by.get((seed, mode))
        counted = counted_attempt(record)
        row = _tracking_row(record, counted)
        # last_reference_index stops at the last tracked search, before the handoff;
        # its tracking-only readings would mislabel handed-off resets, so drop them.
        del row["stalled_before_close_reference"]
        del row["passed_demo_grasp_reference_without_grasp"]
        record = record or {}
        handed = bool(record.get("handed_off"))
        row.update(
            handoff_row=record.get("handoff_row"),
            handed_off=handed,
            handoff_reference_index=record.get("handoff_reference_index"),
            handoff_frame=record.get("handoff_frame"),
            handoff_command=record.get("handoff_command"),
            tracked_commands=record.get("tracked_commands"),
            replayed_commands=record.get("replayed_commands"),
            rollout_parity_checks=record.get("rollout_parity_checks"),
            rollout_parity_mismatches=record.get("rollout_parity_mismatches"),
            ended_before_handoff_without_grasp=counted and not row["grasp"] and not handed,
            handed_off_without_grasp=counted and not row["grasp"] and handed,
        )
        per_reset[str(seed)] = row
    rows = list(per_reset.values())
    counted_attempts = sum(r["counted"] for r in rows)
    result = {
        "grasp_resets": sum(r["grasp"] for r in rows),
        "summed_ordered_stages": sum(r["ordered_stages"] for r in rows),
        "full_successes": sum(r["success"] for r in rows),
        "counted_attempts": counted_attempts,
        "complete": counted_attempts == len(seeds),
        "per_reset": per_reset,
    }
    if mode in HYBRID_PRIVILEGED_MODES:
        mismatches = sum(r["rollout_parity_mismatches"] or 0 for r in rows if r["counted"])
        result.update(
            rollout_parity_mismatches=mismatches,
            rollouts_exact=bool(
                mismatches == 0
                and all(r["rollout_parity_mismatches"] is not None for r in rows if r["counted"])
            ),
            handed_off_resets=sum(r["handed_off"] for r in rows if r["counted"]),
            ended_before_handoff_without_grasp_resets=sum(
                r["ended_before_handoff_without_grasp"] for r in rows
            ),
            handed_off_without_grasp_resets=sum(r["handed_off_without_grasp"] for r in rows),
        )
    return result


def hybrid_ceiling_gate(records, seeds):
    """Preregistered TASK-046 gate; missing/failed attempts count as zero stages."""
    provenance_valid = all(r.get("provenance_valid", True) for r in records)
    primary = _hybrid_arm(records, seeds, "privileged_hybrid")
    early = _hybrid_arm(records, seeds, "privileged_hybrid_early")
    replay = _hybrid_arm(records, seeds, "demo_replay")
    passed = provenance_valid and primary["rollouts_exact"] and primary["grasp_resets"] >= 2
    conclusive = provenance_valid and primary["rollouts_exact"] and primary["complete"]
    early_conclusive = provenance_valid and early["rollouts_exact"] and early["complete"]
    early_grasp = early["grasp_resets"] >= 2
    primary_grasp = primary["grasp_resets"] >= 2
    comparison = (
        "inconclusive: an arm is incomplete, provenance is invalid or rollouts are inexact"
        if not (conclusive and early_conclusive)
        else "both handoffs reach grasp on >=2/4: handoff timing within the last pre-close "
        "horizon is not critical"
        if primary_grasp and early_grasp
        else "only the earlier handoff reaches grasp on >=2/4: tracking over the last pre-close "
        "horizon (or skipping the dead-time descent) is associated with losing the grasp"
        if early_grasp
        else "only the close-row handoff reaches grasp on >=2/4: the longer open-loop segment "
        "from the earlier tracked state is associated with losing the grasp"
        if primary_grasp
        else "neither handoff reaches grasp on >=2/4: open-loop demonstration actions from a "
        "tracked approach state do not reproduce the grasp at either handoff"
    )
    return {
        "label": "NON-LEARNED hybrid phase control under the privileged MuJoCo-rollout ceiling; "
        "any grasp after the handoff is produced by replayed demonstration actions; not a "
        "learned result",
        "privileged_hybrid_grasp_resets": primary["grasp_resets"],
        "summed_ordered_stages": primary["summed_ordered_stages"],
        "full_successes": primary["full_successes"],
        "counted_attempts": primary["counted_attempts"],
        "provenance_valid": provenance_valid,
        "rollout_parity_mismatches": primary["rollout_parity_mismatches"],
        "rollouts_exact": primary["rollouts_exact"],
        "primary_gate_passed": bool(passed),
        "readings": {
            "conclusive": bool(conclusive),
            "tracking_failed_before_handoff": conclusive
            and not passed
            and primary["ended_before_handoff_without_grasp_resets"] >= 3,
            "replay_from_tracked_state_insufficient_for_grasp": conclusive
            and not passed
            and primary["handed_off_without_grasp_resets"] >= 3,
            "interpretation": (
                "tracked approach + open-loop demonstration close reaches grasp under exact "
                "dynamics; the grasp is attributable to the replayed demonstration actions, "
                "not a model"
                if passed and conclusive
                else "tracked approach + open-loop demonstration close does not reach grasp "
                "under exact dynamics"
                if conclusive
                else "inconclusive: missing/failed attempts, invalid provenance or inexact rollouts"
            ),
            "secondary_early_handoff_conclusive": bool(early_conclusive),
            "secondary_early_handoff_grasp_on_two": bool(early_conclusive and early_grasp),
            "handoff_comparison": comparison,
            "demo_replay_conclusive": bool(provenance_valid and replay["complete"]),
        },
        "arms": {
            "privileged_hybrid": primary,
            "privileged_hybrid_early": early,
            "demo_replay": replay,
        },
        "rule": (
            "privileged_hybrid (handoff at the first close row) reaches the unchanged scorer's "
            "grasp stage on >=2 resets with zero rollout parity mismatches in its counted "
            "attempts; only cleanly completed provenance-valid attempts are scored; readings "
            "are conclusive only with all attempts of the arm counted; privileged_hybrid_early "
            "and demo_replay never affect the primary gate; non-learned ceiling only"
        ),
    }


def _object_arm(records, seeds, mode):
    """Per-reset rows for one TASK-047 arm; only counted attempts carry stages."""
    by = {(r["seed"], r["mode"]): r for r in records if r.get("mode") == mode}
    per_reset = {}
    for seed in seeds:
        record = by.get((seed, mode))
        counted = counted_attempt(record)
        score = (record.get("score") or {}) if counted else {}
        record = record or {}
        row = {
            "counted": counted,
            "status": record.get("status"),
            "termination_reason": record.get("termination_reason"),
            "executed_steps": record.get("executed_steps"),
            "demonstration_episode_id": record.get("demonstration_episode_id"),
            "ordered_stages": ordered_stage_count(score),
            "reach": bool(score.get("reach")),
            "grasp": bool(score.get("grasp")),
            "success": bool(score.get("success")),
            "reported_ordered_stages_uncounted": ordered_stage_count(record.get("score")),
        }
        if mode == OBJECT_MODE:
            row.update(
                phase=record.get("phase"),
                furthest_phase_index=record.get("furthest_phase_index"),
                phase_log=record.get("phase_log"),
                rollout_parity_checks=record.get("rollout_parity_checks"),
                rollout_parity_mismatches=record.get("rollout_parity_mismatches"),
                rollout_full_state_parity_checks=record.get("rollout_full_state_parity_checks"),
                rollout_full_state_parity_mismatches=record.get(
                    "rollout_full_state_parity_mismatches"
                ),
            )
        if mode == "scripted_oracle":
            row["oracle_phase"] = record.get("oracle_phase")
        per_reset[str(seed)] = row
    rows = list(per_reset.values())
    counted_attempts = sum(r["counted"] for r in rows)
    result = {
        "grasp_resets": sum(r["grasp"] for r in rows),
        "reach_resets": sum(r["reach"] for r in rows),
        "summed_ordered_stages": sum(r["ordered_stages"] for r in rows),
        "full_successes": sum(r["success"] for r in rows),
        "counted_attempts": counted_attempts,
        "complete": counted_attempts == len(seeds),
        "per_reset": per_reset,
    }
    if mode == OBJECT_MODE:
        counted_rows = [r for r in rows if r["counted"]]
        robot = sum(r["rollout_parity_mismatches"] or 0 for r in counted_rows)
        full = sum(r["rollout_full_state_parity_mismatches"] or 0 for r in counted_rows)
        result.update(
            rollout_parity_mismatches=robot,
            rollout_full_state_parity_mismatches=full,
            # Non-vacuous: every executed command but possibly the last was checked.
            rollouts_exact=bool(
                robot == 0
                and full == 0
                and all(
                    r["rollout_parity_mismatches"] is not None
                    and r["rollout_full_state_parity_mismatches"] is not None
                    and min(
                        r["rollout_parity_checks"] or 0, r["rollout_full_state_parity_checks"] or 0
                    )
                    >= max(0, (r["executed_steps"] or 0) - 1)
                    for r in counted_rows
                )
            ),
            failed_reset_furthest_phase={
                phase: sum(1 for r in counted_rows if not r["grasp"] and r.get("phase") == phase)
                for phase in dict.fromkeys(r.get("phase") for r in counted_rows if not r["grasp"])
            },
        )
    return result


def object_ceiling_gate(records, seeds):
    """Preregistered TASK-047 gate; missing/failed attempts count as zero stages."""
    provenance_valid = all(r.get("provenance_valid", True) for r in records)
    ceiling = _object_arm(records, seeds, OBJECT_MODE)
    replay = _object_arm(records, seeds, "demo_replay")
    scripted = _object_arm(records, seeds, "scripted_oracle")
    c, d, o = ceiling["grasp_resets"], replay["grasp_resets"], scripted["grasp_resets"]
    required = 6 if len(seeds) == 8 else int(np.ceil(0.75 * len(seeds)))
    margin = 3
    exact = ceiling["rollouts_exact"]
    passed = (
        provenance_valid
        and exact
        and ceiling["complete"]
        and replay["complete"]
        and c >= required
        and d <= c - margin
    )
    conclusive = provenance_valid and exact and ceiling["complete"] and replay["complete"]
    scripted_conclusive = provenance_valid and scripted["complete"]
    if not conclusive:
        outcome = "inconclusive"
    elif passed:
        outcome = "separated"
    elif c >= required:
        outcome = "ceiling_adequate_replay_not_separated"
    elif not scripted_conclusive:
        outcome = "ceiling_inadequate_feasibility_reference_inconclusive"
    elif o >= required:
        outcome = "ceiling_inadequate_task_feasible"
    else:
        outcome = "ceiling_inadequate_scripted_also_fails"
    readings = {
        "separated": (
            "the object-aware ceiling grasps and lifts on the wide-jitter resets while open-loop "
            "replay does not: the distribution distinguishes object-aware control from replay "
            "and the phase cost is adequate under exact dynamics",
            "T2: collect the wide-jitter TRAIN data (~200 episodes, >=96 px, dense grasp-phase "
            "branches) under its own preregistration",
        ),
        "ceiling_adequate_replay_not_separated": (
            "the object-aware ceiling is adequate but open-loop replay also grasps: this "
            "jitter does not separate a world model from replay",
            "preregister a wider distribution (larger apple jitter and/or plate jitter) and "
            "repeat this two-arm test before T2",
        ),
        "ceiling_inadequate_task_feasible": (
            "the scripted collector grasps where the object-aware exact-dynamics ceiling does "
            "not: the phase cost/planner design, not the distribution, is inadequate",
            "diagnose the ceiling's failure phases and redesign its cost/transitions under a "
            "new preregistration; do not pair it with a learned model; T2 may proceed only "
            "with the wide distribution validated by the scripted reference",
        ),
        "ceiling_inadequate_scripted_also_fails": (
            "neither the object-aware ceiling nor the scripted collector grasps on >= "
            f"{required} resets: the distribution exceeds the current grasp mechanics",
            "preregister a narrower wide-jitter distribution (e.g. apple +-2 cm) and repeat",
        ),
        "ceiling_inadequate_feasibility_reference_inconclusive": (
            "the object-aware ceiling is inadequate; the scripted feasibility reference is "
            "incomplete, so the cause is not attributed",
            "repair the reference arm and repeat under a new output directory",
        ),
        "inconclusive": (
            "inconclusive: missing/failed ceiling or replay attempts, invalid provenance or "
            "inexact rollouts",
            "fix the software defect and rerun under a new versioned output directory; keep "
            "this run as a recorded failure",
        ),
    }
    interpretation, next_step = readings[outcome]
    return {
        "label": "NON-LEARNED: object-aware privileged MuJoCo-rollout ceiling vs open-loop "
        "retrieved TRAIN demo_replay vs scripted collector reference on the wide-jitter "
        "development resets; not a learned result",
        "privileged_object_grasp_resets": c,
        "demo_replay_grasp_resets": d,
        "scripted_oracle_grasp_resets": o,
        "required_ceiling_grasp_resets": required,
        "required_replay_margin": margin,
        "provenance_valid": provenance_valid,
        "rollouts_exact": bool(exact),
        "primary_gate_passed": bool(passed),
        "readings": {
            "conclusive": bool(conclusive),
            "outcome": outcome,
            "interpretation": interpretation,
            "preregistered_next_step": next_step,
            "scripted_reference_conclusive": bool(scripted_conclusive),
        },
        "arms": {OBJECT_MODE: ceiling, "demo_replay": replay, "scripted_oracle": scripted},
        "rule": (
            "privileged_object reaches the unchanged scorer's grasp stage (reach, >=5 cm lift, "
            "hand contact) on >=6 of 8 resets AND demo_replay grasp resets <= privileged_object "
            "grasp resets - 3; every ceiling and demo_replay attempt counted; zero robot and "
            "full-state rollout parity mismatches over >= executed commands - 1 checks per "
            "counted ceiling attempt; only cleanly completed provenance-valid attempts are "
            "scored; scripted_oracle never affects the gate; non-learned only"
        ),
    }


def gate_summary(plan, records):
    """A privileged ceiling report never carries the learned state gate, and vice versa."""
    if plan.get("goal_kind") == "object":
        return {"object_ceiling_gate": object_ceiling_gate(records, plan["seeds"])}
    if plan.get("goal_kind") == "hybrid":
        return {"hybrid_ceiling_gate": hybrid_ceiling_gate(records, plan["seeds"])}
    if plan.get("goal_kind") == "trajectory":
        if plan.get("privileged_ceiling"):
            return {"tracking_ceiling_gate": tracking_ceiling_gate(records, plan["seeds"])}
        return {"tracking_gate": tracking_gate(records, plan["seeds"])}
    if plan.get("privileged_ceiling"):
        return {"ceiling_gate": ceiling_gate(records, plan["seeds"])}
    if plan.get("goal_kind") == "state":
        return {"state_gate": state_gate(records, plan["seeds"])}
    return {}


def verify_plan_inputs(plan, *, source_root=None):
    source_root = ROOT if source_root is None else Path(source_root)
    if digest(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ContractError("frozen checkpoint changed")
    if digest(Path(plan["dataset"]) / "meta/jepa_manifest.json") != plan["dataset_manifest_sha256"]:
        raise ContractError("frozen dataset manifest changed")
    manifest = json.loads((Path(plan["dataset"]) / "meta/jepa_manifest.json").read_text())
    for relative, expected in manifest.get("sha256", {}).items():
        if digest(Path(plan["dataset"]) / relative) != expected:
            raise ContractError("frozen dataset contents changed")
    for relative, expected in plan["source_hashes"].items():
        if digest(source_root / relative) != expected:
            raise ContractError("frozen evaluation source changed")
    for relative, key in (
        ("configs/g1_sim_action.json", "action_manifest_sha256"),
        ("assets/manifest.json", "asset_manifest_sha256"),
    ):
        if digest(source_root / relative) != plan[key]:
            raise ContractError("frozen action or asset manifest changed")
    assets = json.loads((source_root / "assets/manifest.json").read_text())
    for relative, expected in assets.get("sha256", {}).items():
        if digest(source_root / "third_party/unitree_mujoco" / relative) != expected:
            raise ContractError("frozen upstream asset contents changed")


def prepare_worker(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    verify_plan_inputs(plan)
    if plan.get("goal_kind") in ("state", "trajectory", "hybrid", "object"):
        store, model = checkpoint_model(plan["dataset"], plan["checkpoint"], backend=STATE_BACKEND)
        demonstrations = nominal_calibration_episodes(store)
        arrays, calibration = build_state_goal_library(
            model,
            demonstrations,
            stride=plan["waypoint_stride"],
            dwell=plan["waypoint_dwell"],
            training_episode_ids=store.manifest["splits"]["train"],
        )
        np.savez_compressed(output / "state_goals.npz", **arrays)
        write(output / "state_calibration.json", calibration)
        resolved = plan | {
            "state_goals_sha256": digest(output / "state_goals.npz"),
            "state_calibration_sha256": digest(output / "state_calibration.json"),
            "candidate_demonstration_ids": calibration["calibration_training_episode_ids"],
        }
        if plan.get("goal_kind") in ("trajectory", "hybrid"):
            references, tracking = build_tracking_references(
                model,
                demonstrations,
                calibration,
                fraction=plan["keyframe_threshold_fraction"],
                training_episode_ids=store.manifest["splits"]["train"],
            )
            np.savez_compressed(output / "tracking_references.npz", **references)
            write(output / "tracking_calibration.json", tracking)
            resolved |= {
                "tracking_references_sha256": digest(output / "tracking_references.npz"),
                "tracking_calibration_sha256": digest(output / "tracking_calibration.json"),
            }
        write(output / "resolved_plan.json", resolved)
        return
    store, model = checkpoint_model(plan["dataset"], plan["checkpoint"])
    demo = select_demonstration(store)
    waypoints, calibration = calibrate_waypoints(
        model,
        demo,
        stride=plan["waypoint_stride"],
        dwell=plan["waypoint_dwell"],
        horizon=plan["controller"]["horizon"],
        candidates=plan["controller"]["candidates"],
        proposals=plan["demonstration_proposals"],
        calibration_episodes=nominal_calibration_episodes(store),
        training_episode_ids=store.manifest["splits"]["train"],
    )
    payload = {"images": np.concatenate([w.images["onboard_rgb"] for w in waypoints])}
    for index, waypoint in enumerate(waypoints):
        if waypoint.action_proposals is not None:
            payload[f"proposals_{index}"] = waypoint.action_proposals
    np.savez_compressed(output / "waypoints.npz", **payload)
    write(output / "calibration.json", calibration)
    resolved = plan | {
        "waypoints_sha256": digest(output / "waypoints.npz"),
        "calibration_sha256": digest(output / "calibration.json"),
        "demonstration_episode_id": demo.episode_id,
    }
    write(output / "resolved_plan.json", resolved)


def load_waypoints(output, plan):
    for name, field in (
        ("waypoints.npz", "waypoints_sha256"),
        ("calibration.json", "calibration_sha256"),
    ):
        if digest(output / name) != plan[field]:
            raise ContractError("frozen waypoint artifact changed")
    calibration = json.loads((output / "calibration.json").read_text())
    with np.load(output / "waypoints.npz", allow_pickle=False) as arrays:
        return [
            ImageWaypoint(
                {"onboard_rgb": arrays["images"][i : i + 1]},
                row["threshold"],
                plan["waypoint_dwell"],
                row["source_id"],
                action_proposals=arrays[f"proposals_{i}"] if f"proposals_{i}" in arrays else None,
            )
            for i, row in enumerate(calibration["waypoints"])
        ]


def project_open_loop(robot, requested, *, guarded):
    """Project one open-loop command. In TASK-047 object plans (``guarded``) the unchanged
    joint-velocity guard's refusal is a clean, counted stop (returns None), exactly as
    the object-aware ceiling treats it; every other error, and every refusal in the
    earlier plans, still propagates as a runtime error."""
    from embodied_jepa.object_ceiling import GUARD_REFUSALS

    try:
        return robot.project_candidates(requested[None, None, None])
    except ContractError as error:
        if guarded and str(error) in GUARD_REFUSALS:
            return None
        raise


def attempt_worker(output, attempt_id, *, attempt_seconds=None):
    validate_attempt_budget(attempt_seconds)
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation
    from embodied_jepa.task import AppleToPlateTask

    output = Path(output)
    plan = json.loads((output / "resolved_plan.json").read_text())
    trial = next(t for t in plan["attempts"] if t["attempt_id"] == attempt_id)
    folder = output / "attempts" / attempt_id
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "started.json", trial | {"start_unix_time": time.time()})
    robot = None
    score, failure, reason, steps = {"success": False}, "", "step_limit", 0
    current = {}
    stage = "load"
    start, start_mono = time.time(), time.monotonic()

    def attempt_expired():
        if attempt_seconds is None:
            return False
        elapsed = max(0.0, time.time() - ENTRY_CLOCK[0], time.monotonic() - ENTRY_CLOCK[1])
        return elapsed >= attempt_seconds - min(5.0, attempt_seconds / 10)

    def record(row):
        with (folder / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

    hybrid_kind = plan.get("goal_kind") == "hybrid"
    object_kind = plan.get("goal_kind") == "object"
    tracking_kind = plan.get("goal_kind") in ("trajectory", "hybrid")
    state_kind = plan.get("goal_kind") in ("state", "trajectory", "hybrid", "object")
    privileged = None
    hybrid = None
    object_controller = None
    oracle = None
    progress = {}
    try:
        verify_plan_inputs(plan)
        config = WaypointConfig(
            **(
                plan["controller"]
                | {
                    "seed": trial["seed"],
                    "ablation": trial["mode"]
                    if trial["mode"] in MODES[:3]
                    else (
                        "learned"
                        if trial["mode"] in (PRIVILEGED_MODE,) + HYBRID_PRIVILEGED_MODES
                        else "persistence"
                    ),
                }
            )
        )
        if state_kind:
            store, model = checkpoint_model(
                plan["dataset"], plan["checkpoint"], backend=STATE_BACKEND
            )
            library = load_state_library(output, plan)
            references = load_tracking_references(output, plan) if tracking_kind else None
            controller = None  # Built after initial-RGB retrieval at reset.
        else:
            store, model = checkpoint_model(plan["dataset"], plan["checkpoint"])
            waypoints = load_waypoints(output, plan)
            controller = WaypointController(
                model, waypoints, progress_distance=model.observed_distance, config=config
            )
        stage = "construct"
        robot = G1Embodiment(
            MuJoCoSimulation(object_kind="apple", container_kind="plate", width=96, height=96)
        )
        if robot.state_schema != store.state_schema or json_hash(robot.manifest) != json_hash(
            store.manifest["action_manifest"]
        ):
            raise ContractError(
                "evaluation embodiment disagrees with dataset state/action contracts"
            )
        stage = "reset"
        robot.reset(**trial["reset"])
        scorer = AppleToPlateTask(robot)
        replay = None
        if state_kind:
            stage = "retrieve"
            demo, retrieval = retrieve_demonstration(model, robot.observe().images, library)
            progress.update(
                demonstration_episode_id=demo["episode_id"],
                goal_count=len(demo["goals"]),
                first_close_goal_index=demo["first_close_goal_index"],
            )
            write(folder / "retrieval.json", retrieval | {"goal_frames": demo["goal_frames"]})

            def state_controller(planning_model):
                if not tracking_kind:
                    return WaypointController(
                        planning_model,
                        state_waypoints(library, demo),
                        progress_distance=model.observed_distance,
                        config=config,
                    )
                from embodied_jepa.trajectory_tracking import TrackingController, TrackingRule

                reference, row = tracking_reference(references, demo)
                progress.update(
                    reference_length=row["reference_length"],
                    first_close_reference_index=row["first_close_reference_index"],
                    demo_grasp_reference_index=row["demo_grasp_reference_index"],
                    last_reference_index=0,
                )
                return TrackingController(
                    planning_model,
                    reference,
                    progress_distance=model.observed_distance,
                    config=config,
                    rule=TrackingRule(
                        window=plan["tracking_window"],
                        stall_commands=plan["tracking_stall_commands"],
                        stall_min_advance=plan["tracking_stall_min_advance"],
                    ),
                )

            if trial["mode"] == "demo_replay":
                replay = library["arrays"][f"actions_{demo['index']}"]
            elif trial["mode"] in OBJECT_ONLY_MODES and not object_kind:
                raise ContractError("object-aware modes require a declared object plan")
            elif trial["mode"] == "scripted_oracle":
                from embodied_jepa.scripted import apple_collector_policy

                # NON-LEARNED privileged reference: the collector's initial-truth script.
                oracle = apple_collector_policy(robot.sim.task_truth())
            elif trial["mode"] == OBJECT_MODE:
                if plan.get("privileged_ceiling") is not True:
                    raise ContractError("the object-aware ceiling requires a declared ceiling plan")
                from embodied_jepa.object_ceiling import (
                    ObjectCeilingConfig,
                    ObjectCeilingController,
                    ObjectRolloutModel,
                )

                # NON-LEARNED ceiling: exact rollouts scored by simulator object state.
                privileged = ObjectRolloutModel(model, robot, acknowledge_privileged_ceiling=True)
                progress.update(
                    rollout_parity_checks=0,
                    rollout_parity_mismatches=0,
                    rollout_full_state_parity_checks=0,
                    rollout_full_state_parity_mismatches=0,
                )
                object_controller = ObjectCeilingController(
                    privileged,
                    robot,
                    ObjectCeilingConfig(**plan["object_ceiling"], seed=trial["seed"]),
                    acknowledge_privileged_ceiling=True,
                )
                controller = object_controller
            elif trial["mode"] == PRIVILEGED_MODE:
                if plan.get("privileged_ceiling") is not True:
                    raise ContractError("privileged rollouts require a declared ceiling plan")
                from embodied_jepa.privileged_rollout import PrivilegedRolloutModel

                # NON-LEARNED ceiling: exact simulator rollouts replace the forward model.
                privileged = PrivilegedRolloutModel(
                    model, robot, acknowledge_privileged_ceiling=True
                )
                progress.update(rollout_parity_checks=0, rollout_parity_mismatches=0)
                controller = state_controller(privileged)
            elif trial["mode"] in HYBRID_PRIVILEGED_MODES:
                if not hybrid_kind or plan.get("privileged_ceiling") is not True:
                    raise ContractError("hybrid phase control requires a declared hybrid ceiling")
                from embodied_jepa.hybrid_phase import HybridPhaseController, handoff_row
                from embodied_jepa.privileged_rollout import PrivilegedRolloutModel

                # NON-LEARNED: exact rollouts before the handoff, demo actions after it.
                privileged = PrivilegedRolloutModel(
                    model, robot, acknowledge_privileged_ceiling=True
                )
                progress.update(rollout_parity_checks=0, rollout_parity_mismatches=0)
                tracker = state_controller(privileged)
                hybrid = HybridPhaseController(
                    tracker,
                    library["arrays"][f"actions_{demo['index']}"],
                    handoff_row=handoff_row(
                        progress["first_close_reference_index"],
                        plan["hybrid_handoff_rows_before_close"][trial["mode"]],
                    ),
                    lower_bounds=config.lower_bounds,
                    upper_bounds=config.upper_bounds,
                )
                controller = hybrid
            else:
                controller = state_controller(model)
        rng = np.random.default_rng(trial["seed"])
        last_grasps = np.array(config.initial_grasps, np.float32)
        for step in range(config.max_steps):
            if attempt_expired():
                reason = "attempt_timeout"
                record({"event": "attempt_timeout", "executed": False, "stage": "before_observe"})
                break
            control_start_wall, control_start_mono = time.time(), time.monotonic()
            stage = "observe"
            observation = robot.observe()
            stage = "plan"
            if trial["mode"] in CONTROLLER_MODES + HYBRID_PRIVILEGED_MODES + (OBJECT_MODE,):
                decision = controller.step(observation, robot.project_candidates)
                if tracking_kind:
                    progress["last_reference_index"] = decision.trace.get(
                        "reference_index", progress.get("last_reference_index")
                    )
                elif state_kind and not object_kind:
                    progress["last_goal_index"] = decision.trace.get(
                        "goal_index", progress.get("last_goal_index")
                    )
                if decision.action is None:
                    reason = decision.termination_reason
                    if state_kind:
                        record(decision.trace | {"event": "controller_termination"})
                    break
                command, current = decision.action, decision.trace.copy()
                if privileged is not None:
                    current["privileged_rollout_diagnostic"] = privileged.pop_diagnostics()
                    for row in current["privileged_rollout_diagnostic"]:
                        match = row.get("previous_search_first_step_exact_match")
                        if match is not None:
                            progress["rollout_parity_checks"] = (
                                progress.get("rollout_parity_checks", 0) + 1
                            )
                            progress["rollout_parity_mismatches"] = progress.get(
                                "rollout_parity_mismatches", 0
                            ) + int(not match)
                        full = row.get("previous_search_first_step_full_state_exact_match")
                        if full is not None:
                            progress["rollout_full_state_parity_checks"] += 1
                            progress["rollout_full_state_parity_mismatches"] += int(not full)
                elif state_kind:
                    current["visual_diagnostic"] = model.pop_diagnostics()
            elif trial["mode"] == "demo_replay":
                # NON-LEARNED reference: recorded TRAIN actions, open loop, no model call.
                if step >= len(replay):
                    reason = "demo_exhausted"
                    break
                requested = np.clip(replay[step], config.lower_bounds, config.upper_bounds).astype(
                    np.float32
                )
                projected = project_open_loop(robot, requested, guarded=object_kind)
                if projected is None:
                    reason = "guard_refused"
                    record({"event": "guard_refused", "step": step, "executed": False})
                    break
                if not projected.feasible[0, 0]:
                    raise ContractError("demo replay command has no feasible projection")
                command = projected.actions[0, 0, 0]
                current = {
                    "step": step,
                    "control": "demo_replay_non_learned",
                    "replay_frame": step,
                    "recorded_action": replay[step].tolist(),
                    "sampled_action": requested.tolist(),
                    "projected_action": command.tolist(),
                    "observation_timestamp": float(observation.timestamps[0]),
                    "goal_index": None,
                }
            elif trial["mode"] == "scripted_oracle":
                # NON-LEARNED privileged reference: collector script, same bounds/projection.
                if oracle.done:
                    reason = "policy_complete"
                    break
                phase = oracle.phase
                proposed = oracle.action(robot)
                requested = np.clip(proposed, config.lower_bounds, config.upper_bounds).astype(
                    np.float32
                )
                projected = project_open_loop(robot, requested, guarded=True)
                if projected is None:
                    reason = "guard_refused"
                    record({"event": "guard_refused", "step": step, "executed": False})
                    break
                if not projected.feasible[0, 0]:
                    raise ContractError("scripted command has no feasible projection")
                command = projected.actions[0, 0, 0]
                current = {
                    "step": step,
                    "control": "scripted_oracle_non_learned",
                    "oracle_phase": phase,
                    "proposed_action": proposed.tolist(),
                    "sampled_action": requested.tolist(),
                    "projected_action": command.tolist(),
                    "observation_timestamp": float(observation.timestamps[0]),
                    "goal_index": None,
                }
            else:
                requested = np.zeros(14, np.float32)
                requested[12:] = last_grasps
                if trial["mode"] == "random":
                    requested = rng.uniform(config.lower_bounds, config.upper_bounds).astype(
                        np.float32
                    )
                projected = robot.project_candidates(requested[None, None, None])
                if not projected.feasible[0, 0]:
                    raise ContractError("baseline command has no feasible projection")
                command = projected.actions[0, 0, 0]
                current = {
                    "step": step,
                    "sampled_action": requested.tolist(),
                    "projected_action": command.tolist(),
                    "observation_timestamp": float(observation.timestamps[0]),
                    "goal_index": None,
                }
            control_elapsed = max(
                time.time() - control_start_wall, time.monotonic() - control_start_mono
            )
            current["control_seconds"] = control_elapsed
            if control_elapsed > plan["control_timeout_seconds"]:
                reason = "deadline_miss"
                record(current | {"event": "deadline_miss", "executed": False, "reason": reason})
                break
            if attempt_expired():
                reason = "attempt_timeout"
                record(
                    current
                    | {"event": "attempt_timeout", "executed": False, "stage": "before_execute"}
                )
                break
            current["checkpoint_sha256"] = plan.get("checkpoint_sha256")
            record(current | {"event": "command_pending", "executed": None})
            stage = "execute"
            result = robot.execute(command)
            if oracle is not None:
                oracle.advance(result)
            if trial["mode"] in CONTROLLER_MODES + HYBRID_PRIVILEGED_MODES + (OBJECT_MODE,):
                diagnostic = current.get("visual_diagnostic")
                rollout = current.get("privileged_rollout_diagnostic")
                current = controller.acknowledge(result)
                if privileged is not None:
                    current["privileged_rollout_diagnostic"] = rollout
                elif state_kind:
                    current["visual_diagnostic"] = diagnostic  # weight 0, logged only
            else:
                current.update(
                    status=result.status,
                    reason=result.reason,
                    applied_action=None
                    if result.applied_action is None
                    else result.applied_action.tolist(),
                    execution_timestamp=result.timestamp,
                )
            current["control_seconds"] = control_elapsed
            current["checkpoint_sha256"] = plan.get("checkpoint_sha256")
            accepted = result.applied_action is not None
            steps += int(accepted)
            if accepted:
                last_grasps = result.applied_action[12:].copy()
            stage = "score"
            score = scorer.evaluate()
            record(current | {"event": "result", "executed": accepted, "score": score})
            if not accepted:
                reason = "execution_rejected"
                break
            if score["success"]:
                reason = "success"
                break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        reason = "runtime_error"
        record(
            {
                "event": "error",
                "stage": stage,
                "error": failure,
                "execution_uncertain": stage == "execute",
                "partial_trace": current,
            }
        )
    finally:
        if privileged is not None:
            privileged.close()
        if robot is not None:
            robot.stop(reason)
            robot.close()
    if state_kind:
        progress["ordered_stage_count"] = ordered_stage_count(score)
    if hybrid is not None:
        progress.update(hybrid.summary())
    if object_controller is not None:
        progress.update(object_controller.summary())
    if oracle is not None:
        progress.update(oracle_phase=oracle.phase, oracle_commands=oracle.step_count)
    report = (
        trial
        | progress
        | {
            "status": "attempt_timeout" if reason == "attempt_timeout" else "completed",
            "attempt_allocated_seconds": attempt_seconds,
            "termination_reason": reason,
            "error": failure,
            "executed_steps": steps,
            "score": score,
            "success": bool(score["success"]),
            "wall_seconds": max(0.0, time.time() - start, time.monotonic() - start_mono)
            if attempt_seconds is None
            else max(0.0, time.time() - ENTRY_CLOCK[0], time.monotonic() - ENTRY_CLOCK[1]),
        }
    )
    write(folder / "report.json", report)


def supervise(command, *, cwd, log, remaining, popen=subprocess.Popen, sleep=time.sleep):
    """Hard wall supervision includes suspension; kill the isolated child process group."""
    start_wall, start_mono = time.time(), time.monotonic()
    with Path(log).open("w") as stream:
        process = popen(
            command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        while True:
            returncode = process.poll()
            elapsed = max(time.time() - start_wall, time.monotonic() - start_mono)
            if elapsed >= remaining:
                if returncode is None:
                    try:
                        os.killpg(process.pid, 9)
                    except ProcessLookupError:
                        pass
                    process.wait()
                return {"status": "hard_wall_timeout", "returncode": process.returncode}
            if returncode is not None:
                return {"status": "exited", "returncode": returncode}
            sleep(min(0.1, max(0.001, remaining - elapsed)))


def summarize_attempt(output, trial, fallback):
    folder = Path(output) / "attempts" / trial["attempt_id"]
    if (folder / "report.json").exists():
        return json.loads((folder / "report.json").read_text())
    results, pending = [], False
    if (folder / "trace.jsonl").exists():
        for line in (folder / "trace.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A killed writer can leave an incomplete final JSON line.
            if row.get("event") == "command_pending":
                pending = True
            elif row.get("event") == "result":
                results.append(row)
                pending = False
    return trial | {
        "status": fallback,
        "success": False,
        "executed_steps": sum(r.get("executed") is True for r in results),
        "execution_uncertain": pending,
        "score": results[-1].get("score", {}) if results else {},
        "started": (folder / "started.json").exists(),
    }


def clean_completion(records):
    return bool(records) and all(
        r.get("status") == "completed"
        and r.get("termination_reason") not in ("runtime_error", "deadline_miss", "attempt_timeout")
        and r.get("provenance_valid", True)
        for r in records
    )


def verify_generated(output, resolved, expected_digest):
    if digest(output / "resolved_plan.json") != expected_digest:
        raise ContractError("resolved evaluation plan changed")
    names = (
        (
            ("state_goals.npz", "state_goals_sha256"),
            ("state_calibration.json", "state_calibration_sha256"),
        )
        if resolved.get("goal_kind") in ("state", "trajectory", "hybrid", "object")
        else (("waypoints.npz", "waypoints_sha256"), ("calibration.json", "calibration_sha256"))
    )
    if resolved.get("goal_kind") in ("trajectory", "hybrid"):
        names += (
            ("tracking_references.npz", "tracking_references_sha256"),
            ("tracking_calibration.json", "tracking_calibration_sha256"),
        )
    for name, key in names:
        if digest(output / name) != resolved[key]:
            raise ContractError("frozen waypoint calibration changed")


def run(args, *, start_clock=None):
    start_wall, start_mono = start_clock or (time.time(), time.monotonic())
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("evaluation output must be new")
    plan = make_plan(
        stage=args.stage,
        seeds=args.seeds,
        modes=args.modes,
        max_seconds=args.max_seconds,
        max_steps=args.max_steps,
        stride=args.stride,
        dwell=args.dwell,
        horizon=args.horizon,
        candidates=args.candidates,
        iterations=args.iterations,
        proposals=not args.no_proposals,
        control_timeout=args.control_timeout,
        commitment_steps=getattr(args, "commitment_steps", 1),
        attempt_max_seconds=getattr(args, "attempt_max_seconds", None),
        goal_kind=getattr(args, "goal_kind", "image"),
        goal_stall_limit=getattr(args, "goal_stall_limit", None),
    )
    if args.stage == "final" and args.selection is None:
        raise ValueError("final cohort requires an explicit frozen development-selection JSON")
    dataset, checkpoint = Path(args.dataset).resolve(), Path(args.checkpoint).resolve()
    plan.update(
        dataset=str(dataset),
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        dataset_manifest_sha256=digest(dataset / "meta/jepa_manifest.json"),
    )
    if args.selection is not None:
        selection = json.loads(Path(args.selection).read_text())
        for key in ("checkpoint_sha256", "dataset_manifest_sha256"):
            if selection.get(key) != plan[key]:
                raise ValueError("development selection does not match frozen input hashes")
        for key in (
            "waypoint_stride",
            "waypoint_dwell",
            "demonstration_proposals",
            "control_timeout_seconds",
        ):
            if selection.get(key) != plan[key]:
                raise ValueError("development selection waypoint protocol mismatch")
        if selection.get("controller") != json.loads(json.dumps(plan["controller"])):
            raise ValueError("development selection does not match controller configuration")
        plan["selection_sha256"] = digest(args.selection)
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    shutil.copytree(
        ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (snapshot / "scripts").mkdir()
    shutil.copyfile(Path(__file__), snapshot / "scripts/evaluate_apple.py")
    for directory, filename in (("configs", "g1_sim_action.json"), ("assets", "manifest.json")):
        (snapshot / directory).mkdir()
        shutil.copyfile(ROOT / directory / filename, snapshot / directory / filename)
    (snapshot / "third_party").symlink_to(ROOT / "third_party", target_is_directory=True)
    plan["source_hashes"] = {
        str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))
    }
    plan["action_manifest_sha256"] = digest(snapshot / "configs/g1_sim_action.json")
    plan["asset_manifest_sha256"] = digest(snapshot / "assets/manifest.json")
    plan["source_revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    write(output / "plan.json", plan)

    def remaining():
        return (
            plan["max_seconds"]
            - plan["finalization_reserve_seconds"]
            - max(time.time() - start_wall, time.monotonic() - start_mono)
        )

    worker = [
        sys.executable,
        str(snapshot / "scripts/evaluate_apple.py"),
        "--worker-output",
        str(output),
    ]
    preparation = (
        {"status": "not_started_budget", "returncode": None}
        if remaining() <= 0
        else supervise(
            worker + ["--worker", "prepare"],
            cwd=snapshot,
            log=output / "prepare.log",
            remaining=remaining(),
        )
    )
    records = []
    ready = (
        preparation["status"] == "exited"
        and preparation["returncode"] == 0
        and (output / "resolved_plan.json").exists()
    )
    resolved = json.loads((output / "resolved_plan.json").read_text()) if ready else None
    resolved_digest = digest(output / "resolved_plan.json") if ready else None
    invalid = ""
    for trial in plan["attempts"]:
        outcome = None
        allocation = None
        fallback = "preparation_failed" if not ready else "not_started_budget"
        if invalid:
            fallback = "not_started_invalid_inputs"
        elif ready and remaining() > 0:
            try:
                verify_plan_inputs(plan, source_root=snapshot)
                verify_generated(output, resolved, resolved_digest)
                if remaining() <= 0:
                    fallback = "not_started_budget"
                else:
                    available = remaining()
                    requested_cap = plan["attempt_max_seconds"]
                    allocated = (
                        available if requested_cap is None else min(requested_cap, available)
                    )
                    allocation = {
                        "requested_seconds": requested_cap,
                        "allocated_seconds": allocated,
                        "global_remaining_seconds": available,
                        "global_shortened": requested_cap is not None and allocated < requested_cap,
                    }
                    command = worker + ["--worker", "attempt"]
                    if requested_cap is not None:
                        command += ["--worker-attempt-seconds", str(allocated)]
                    command += ["--attempt-id", trial["attempt_id"]]
                    outcome = supervise(
                        command,
                        cwd=snapshot,
                        log=output / f"{trial['attempt_id']}.log",
                        remaining=allocated,
                    )
                    fallback = outcome["status"] if outcome["returncode"] else "missing_report"
                    verify_plan_inputs(plan, source_root=snapshot)
                    verify_generated(output, resolved, resolved_digest)
            except (ContractError, OSError, ValueError) as error:
                invalid = f"{type(error).__name__}: {error}"
                fallback = "invalid_inputs"
        record = summarize_attempt(output, trial, fallback)
        record["supervisor"] = outcome
        record["attempt_allocation"] = allocation
        if outcome is not None and (outcome["returncode"] != 0 or outcome["status"] != "exited"):
            record.update(
                worker_report_status=record["status"],
                reported_success_before_failure=record["success"],
                success=False,
                status=(
                    "attempt_timeout"
                    if allocation
                    and allocation["requested_seconds"] is not None
                    and not allocation["global_shortened"]
                    else "hard_wall_timeout"
                )
                if outcome["status"] == "hard_wall_timeout"
                else "child_failed",
            )
        if record["status"] in ("attempt_timeout", "hard_wall_timeout"):
            scope = (
                "attempt"
                if allocation
                and allocation["requested_seconds"] is not None
                and not allocation["global_shortened"]
                else "global"
            )
            record["timeout_scope"] = scope
            record["status"] = "attempt_timeout" if scope == "attempt" else "hard_wall_timeout"
            record["success"] = False
        if invalid:
            record.update(status=fallback, provenance_valid=False, provenance_error=invalid)
            for prior in records:
                prior["provenance_valid"] = False
        records.append(record)
        write(
            output / "report.json",
            {
                "plan_sha256": digest(output / "plan.json"),
                "resolved_plan_sha256": resolved_digest,
                "preparation": preparation,
                "planned": len(plan["attempts"]),
                "records": records,
                "status": "completed"
                if len(records) == len(plan["attempts"]) and clean_completion(records)
                else "incomplete",
                "provenance_valid": not bool(invalid),
                "provenance_error": invalid,
                "successes": 0 if invalid else sum(r["success"] for r in records),
                "denominator": len(plan["attempts"]),
                "wall_seconds": max(time.time() - start_wall, time.monotonic() - start_mono),
            }
            | gate_summary(plan, records),
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--modes", choices=MODES, nargs="+")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--attempt-max-seconds", type=float)
    parser.add_argument("--commitment-steps", type=int, default=1)
    parser.add_argument("--control-timeout", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--dwell", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--no-proposals", action="store_true")
    parser.add_argument(
        "--goal-kind",
        choices=("image", "state", "trajectory", "hybrid", "object"),
        default="image",
    )
    parser.add_argument("--goal-stall-limit", type=int)
    parser.add_argument("--worker", choices=("prepare", "attempt"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-attempt-seconds", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    validate_attempt_budget(args.worker_attempt_seconds)
    if args.worker:
        prepare_worker(args.worker_output) if args.worker == "prepare" else attempt_worker(
            args.worker_output, args.attempt_id, attempt_seconds=args.worker_attempt_seconds
        )
        return
    if not all((args.dataset, args.checkpoint, args.output)):
        parser.error("dataset, checkpoint and output are required")
    records = run(args, start_clock=ENTRY_CLOCK)
    return 0 if clean_completion(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
