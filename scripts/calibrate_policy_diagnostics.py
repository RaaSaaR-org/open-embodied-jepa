"""Pre-freeze calibration for amendment 1 of ``apple_policy_diagnostics_v1`` (TASK-057).

Generates every number amendment 1 adds to the protocol and manifest, so that each is
re-derivable from committed code (the protocol's §9 standard). **No trained arm is run in closed
loop here**; the closed-loop part drives only scripted and blind controllers.

``offline`` (decodes the val split, and the train split's reset frames only):
  * step-zero error of every arm on RECORDED reset frames against D1's frozen thresholds -- the
    population D1 is actually scored on, which the frozen headroom check never measured;
  * D1-pipeline float-noise floor and input sensitivity on stored val reset frames;
  * the open-loop first-departure distribution that calibrates D2's persistence rule.

``closed-loop`` (simulator; development seeds; scripted and blind controllers only):
  * the step-zero commands of ``apple_collector_policy`` and plain ``OracleManipulationPolicy``
    on the 16 development resets;
  * ``full`` substitution with each of the two experts (B3 dry run, and what the unamended text
    would have run);
  * every G-SUB candidate with a BLIND complement -- the prior-only mean command and the
    time-indexed mean command of the surviving train roots -- which is what a G-SUB candidate
    scores when the controller contributes nothing.

    uv run --no-sync python scripts/calibrate_policy_diagnostics.py closed-loop \
        --output outputs/task057-diagnostics/calibration-closed-loop.json
    uv run --no-sync python scripts/calibrate_policy_diagnostics.py offline \
        --output outputs/task057-diagnostics/calibration-offline.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import policy_diagnostics as pd  # noqa: E402

PROTOCOL = "apple_policy_diagnostics_v1"
AMENDMENT = 1
CONFIG = ROOT / "configs" / "apple_policy_v1.yaml"
ARMS = {
    "a0": "A0_proprio_only",
    "a1": "A1_random_encoder",
    "a2": "A2_bc_frozen_e0",
    "a3": "A3_bc_finetuned_e0",
}
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json"
MAX_STEPS = 1000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner():
    return _load("_evaluate_policy", ROOT / "scripts" / "evaluate_policy.py")


def collector():
    return _load("_collect_apple_wide", ROOT / "scripts" / "collect_apple_wide.py")


def surviving_train_roots():
    """Seeds of the non-aim-offset TRAIN roots, from the committed collection plan."""
    from embodied_jepa.cloning import AIM_OFFSET_SEEDS

    c = collector()
    splits = c.split_assignment(c.FROZEN_SEEDS)
    return [s for s in c.FROZEN_SEEDS if splits[s] == "train" and s not in AIM_OFFSET_SEEDS]


def blind_tables(seeds):
    """Prior-only mean and time-indexed mean of ``collector__base_action``'s free components.

    Read from the committed label sidecars of the surviving train roots, every command row.
    """
    rows = []
    for seed in seeds:
        labels = np.load(ROOT / "data" / "apple-wide-v1" / "labels" / f"wide-{seed}.npz")
        rows.append(np.asarray(labels["collector__base_action"], np.float32)[:, pd.FREE_INDICES])
    stacked = np.concatenate(rows)
    length = max(len(r) for r in rows)
    table = np.zeros((length, 7), np.float64)
    for t in range(length):
        table[t] = np.mean([r[t] for r in rows if len(r) > t], axis=0)
    return stacked.mean(0).astype(np.float32), table.astype(np.float32), len(stacked)


# ----- closed loop ---------------------------------------------------------------------------
def _robot():
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    return G1Embodiment(
        MuJoCoSimulation(object_kind="apple", container_kind="plate", width=112, height=112)
    )


def _attempt(job):
    """One closed-loop attempt of a scripted or blind controller (worker process)."""
    from embodied_jepa.scripted import OracleManipulationPolicy
    from embodied_jepa.task import AppleToPlateTask

    r = runner()
    seed, configuration, controller_kind, expert_kind, payload = job
    reset = r.cohort_resets((seed,))[seed]
    robot = _robot()
    robot.reset(**reset)
    scorer = AppleToPlateTask(robot)
    truth = robot.sim.task_truth()
    # None = the preregistered, guarded shadow expert; the plain oracle is the dry run of what
    # the frozen text would have run.
    factory = None if expert_kind == "collector" else OracleManipulationPolicy
    shadow = pd.ShadowExpert(truth, factory=factory)
    if controller_kind == "constant":
        controller = pd.ConstantController(payload, name="prior_only_mean")
    elif controller_kind == "time_indexed":
        controller = pd.TimeIndexedController(payload, name="time_indexed_mean")
    else:
        raise ValueError(controller_kind)
    config = __import__("embodied_jepa.config", fromlist=["ExperimentConfig"]).ExperimentConfig
    cfg = config.load(CONFIG)
    result = pd.run_diagnostic_attempt(
        controller,
        robot,
        scorer,
        configuration=configuration,
        clip=r.clip_to_configured_bounds,
        lower=np.asarray(cfg.planner.lower_bounds, np.float32),
        upper=np.asarray(cfg.planner.upper_bounds, np.float32),
        max_steps=MAX_STEPS,
        deadline_seconds=5.0,
        shadow=shadow,
        trace=False,
    )
    result.pop("departures")
    result.update(seed=seed, expert=expert_kind)
    return (configuration, controller_kind, expert_kind), result


def step_zero_commands(seeds):
    from embodied_jepa.scripted import OracleManipulationPolicy

    r = runner()
    resets = r.cohort_resets(seeds)
    out = {}
    for seed in seeds:
        robot = _robot()
        robot.reset(**resets[seed])
        truth = robot.sim.task_truth()
        out[str(seed)] = {
            "apple_collector_policy": pd.collector_shadow_policy(truth)
            .action(robot)[list(pd.FREE_INDICES)]
            .tolist(),
            "OracleManipulationPolicy": OracleManipulationPolicy(truth)
            .action(robot)[list(pd.FREE_INDICES)]
            .tolist(),
        }
    return out


def closed_loop(output, workers):
    from embodied_jepa.training import source_identity

    r = runner()
    seeds = tuple(r.COHORT_D)
    started = time.perf_counter()
    train_seeds = surviving_train_roots()
    mean, table, n_rows = blind_tables(train_seeds)
    jobs = []
    for seed in seeds:
        for expert in ("collector", "plain_oracle"):
            jobs.append((seed, "full", "constant", expert, mean))
        # The blind controllers on their own: what an unsubstituted controller that sees
        # nothing (or only the step index) scores. Context for the pure policy's 1/16.
        jobs.append((seed, "none", "constant", "collector", mean))
        jobs.append((seed, "none", "time_indexed", "collector", table))
        for name in pd.G_SUB_CANDIDATES:
            jobs.append((seed, name, "constant", "collector", mean))
            jobs.append((seed, name, "time_indexed", "collector", table))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_attempt, jobs, chunksize=1)
    grouped = {}
    for key, result in results:
        grouped.setdefault("|".join(key), []).append(result)
    summary = {
        key: {
            "grasp_resets": sum(a["grasp"] for a in attempts),
            "full_successes": sum(a["success"] for a in attempts),
            "attempts": len(attempts),
            "terminations": sorted({a["termination_reason"] for a in attempts}),
            "per_seed": {
                str(a["seed"]): {
                    "grasp": a["grasp"],
                    "success": a["success"],
                    "termination_reason": a["termination_reason"],
                    "executed_steps": a["executed_steps"],
                    "shadow_expert_exhausted_at_step": a["shadow_expert_exhausted_at_step"],
                }
                for a in sorted(attempts, key=lambda a: a["seed"])
            },
        }
        for key, attempts in sorted(grouped.items())
    }
    zero = step_zero_commands(seeds)
    collector_zero = np.array([zero[str(s)]["apple_collector_policy"] for s in seeds])
    plain_zero = np.array([zero[str(s)]["OracleManipulationPolicy"] for s in seeds])
    report = {
        "protocol": PROTOCOL,
        "amendment": AMENDMENT,
        "part": "closed-loop",
        "learned_arms_run": False,
        "cohort": "D_development_never_gating",
        "seeds": list(seeds),
        "source": source_identity(),
        "blind_controllers": {
            "surviving_train_roots": len(train_seeds),
            "command_rows": n_rows,
            "prior_only_mean": mean.tolist(),
            "time_indexed_mean_length": len(table),
        },
        "step_zero": {
            "per_seed": zero,
            "collector_std_per_dimension": collector_zero.std(0).tolist(),
            "median_abs_collector_minus_plain": np.median(
                np.abs(collector_zero - plain_zero), 0
            ).tolist(),
            "seeds_where_the_two_differ": int(
                (np.abs(collector_zero - plain_zero) > 1e-6).any(1).sum()
            ),
        },
        "runs": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write(output, report)
    return report


# ----- provenance ----------------------------------------------------------------------------
#: Two surviving TRAIN roots per perturbation level (level = plan index % 4).
PROVENANCE_ROOTS = (48000, 48008, 48001, 48013, 48002, 48006, 48003, 48011)


def _replay(seed):
    """Re-run the collector's own loop on one root; compare with the stored base actions.

    Returns the max |replayed - stored| for ``apple_collector_policy`` (the collector's own
    loop, perturbations included) and, for plain ``OracleManipulationPolicy`` advanced on the
    SAME executed results, how many recorded commands it would have issued differently.
    """
    from types import SimpleNamespace

    from embodied_jepa.hand_crop import HandCrop
    from embodied_jepa.scripted import OracleManipulationPolicy, apple_collector_policy
    from embodied_jepa.task import AppleToPlateTask

    c = collector()
    root = next(r for r in c.make_plan(c.FROZEN_SEEDS)["roots"] if r["seed"] == seed)
    if root["aim_offset_xy_m"] is not None or root["split"] != "train":
        raise ValueError(f"{seed} is not a surviving train root")
    labels = np.load(ROOT / "data" / "apple-wide-v1" / "labels" / f"wide-{seed}.npz")
    stored = np.asarray(labels["collector__base_action"], np.float32)
    robot = _robot()
    robot.reset(root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"])
    crop = HandCrop(robot, render_size=c.CROP_RENDER_SIZE, crop_size=c.IMAGE_SIZE)
    truth = robot.sim.task_truth()
    controller = c.Controller(
        apple_collector_policy(truth), c.Perturber(root["noise_level"], root["noise_seed"])
    )
    plain = OracleManipulationPolicy(truth)
    scorer = AppleToPlateTask(robot)
    rec = c.Recorder()
    rec.observe(robot, crop, scorer.evaluate())
    replay_error, plain_differs, plain_compared = 0.0, 0, 0
    for t in range(len(stored)):
        plain_action = None if plain.done else plain.action(robot)
        _score, stop = c.step(robot, scorer, crop, controller, rec)
        if stop:
            raise RuntimeError(f"replay of {seed} stopped early at {t}: {stop}")
        base = rec.steps["base"][-1]
        replay_error = max(replay_error, float(np.abs(base - stored[t]).max()))
        if plain_action is not None:
            plain_compared += 1
            plain_differs += int(np.abs(plain_action[6:14] - base[6:14]).max() > 1e-6)
            plain.advance(SimpleNamespace(applied_action=rec.steps["applied"][-1], reason=None))
    return {
        "seed": seed,
        "noise_level": root["noise_level"],
        "stored_commands": len(stored),
        "max_abs_replay_minus_stored": replay_error,
        "plain_oracle_commands_compared": plain_compared,
        "plain_oracle_commands_differing": plain_differs,
        "stored_phase_indices_visited": sorted({int(v) for v in labels["collector__phase_index"]}),
    }


def provenance(output, workers):
    """Which policy produced the BC targets: replay, phase budgets, and the release signature."""
    from collections import Counter

    from embodied_jepa.scripted import OracleManipulationPolicy
    from embodied_jepa.training import source_identity

    started = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        replays = pool.map(_replay, PROVENANCE_ROOTS, chunksize=1)
    c = collector()
    from embodied_jepa.cloning import AIM_OFFSET_SEEDS

    # TEST-split roots are excluded: not even their label sidecars are read here.
    splits = c.split_assignment(c.FROZEN_SEEDS)
    non_aim = [
        s for s in c.FROZEN_SEEDS if s not in AIM_OFFSET_SEEDS and splits[s] in ("train", "val")
    ]
    per_phase = {k: [] for k in range(8)}
    lengths, ramp_values = [], 0
    for seed in non_aim:
        labels = np.load(ROOT / "data" / "apple-wide-v1" / "labels" / f"wide-{seed}.npz")
        phase = np.asarray(labels["collector__phase_index"])
        base = np.asarray(labels["collector__base_action"], np.float32)
        counts = Counter(phase.tolist())
        for k in range(8):
            per_phase[k].append(counts.get(k, 0))
        lengths.append(len(phase))
        # EarlyRelease's opening ramp: grasp base action 1 - 0.08 = 0.92 in phases >= 5.
        ramp_values += int(np.isclose(base[phase >= 5, 13], 0.92, atol=1e-6).sum())
    fake = {
        "position_frame": "world",
        "base_position_world": [0.0, 0.0, 0.0],
        "base_rotation_world": np.eye(3),
        "object_position": [0.34, -0.18, 0.77],
        "plate_position": [0.49, -0.09, 0.75],
        "container_surface_z": 0.76,
        "object_support_height": 0.04,
    }
    collector_policy = pd.collector_shadow_policy(fake)
    plain = OracleManipulationPolicy(fake)
    report = {
        "protocol": PROTOCOL,
        "amendment": AMENDMENT,
        "part": "provenance",
        "source": source_identity(),
        "budgets": {
            "apple_collector_policy": {
                "phases": [p.name for p in collector_policy.phases],
                "commands": [p.commands for p in collector_policy.phases],
                "total": collector_policy.max_steps,
            },
            "OracleManipulationPolicy": {
                "phases": [p.name for p in plain.phases],
                "commands": [p.commands for p in plain.phases],
                "total": plain.max_steps,
            },
            "collect_apple_wide_ROOT_MAX_COMMANDS": c.ROOT_MAX_COMMANDS,
        },
        "corpus_non_aim_train_and_val_roots": len(non_aim),
        "test_split_roots_read": 0,
        "corpus_max_root_commands": max(lengths),
        "corpus_phase_commands_per_root": {
            str(k): {
                "mean_over_all_non_aim_roots": float(np.mean(v)),
                "roots_visiting": int(sum(1 for x in v if x)),
                "max": int(max(v)),
                "distinct_nonzero": sorted({int(x) for x in v if x}),
            }
            for k, v in per_phase.items()
        },
        "corpus_opening_ramp_rows_grasp_0_92_in_phase_5_plus": ramp_values,
        "replays": replays,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write(output, report)
    return report


# ----- offline -------------------------------------------------------------------------------
def offline(output, workers, device):
    import torch

    from embodied_jepa.cloning import is_surviving_root, load_bc_split, precompute_features
    from embodied_jepa.policy import FREE_ACTION_INDICES, FREE_ACTION_NAMES
    from embodied_jepa.training import source_identity
    from embodied_jepa.world_model_v2 import _open, images

    measure = _load("_measure", ROOT / "scripts" / "measure_policy_offline_conditionals.py")
    manifest = json.loads(MANIFEST.read_text())
    thresholds = manifest["frozen_D1_thresholds_three_times_table_A"]
    table_a = manifest["frozen_offline_error_table_A"]["values"]
    names = list(FREE_ACTION_NAMES)
    arm_names = [f"right_{d}" for d in pd.ARM_DIMENSIONS]
    started = time.perf_counter()
    config, store, _settings, cameras = _open(CONFIG)
    report = {
        "protocol": PROTOCOL,
        "amendment": AMENDMENT,
        "part": "offline",
        "device": device,
        "source": source_identity(),
        "step_zero": {},
        "d1_pipeline_calibration": {},
        "d2_departure_calibration": {},
    }
    for split in ("val", "train"):
        arrays, rows, _counts, base = load_bc_split(
            store, split, cameras, workers=workers, limit=None, acknowledge=True
        )
        ids = [e for e in arrays.episode_ids if is_surviving_root(e)]
        first = np.array([int(arrays.offsets[arrays.episode_ids.index(e)]) for e in ids], np.int64)
        if not np.isin(first, rows).all():
            raise RuntimeError("a surviving root's reset frame is not a sampled BC row")
        targets = base[first][:, list(FREE_ACTION_INDICES)]
        prior = np.median(targets, 0)
        entry = {
            "reset_frames": len(first),
            "state_max_std_across_resets": float(arrays.states[first].std(0).max()),
            "target_std_per_dimension": dict(zip(names, targets.std(0).tolist(), strict=True)),
            "target_median_per_dimension": dict(zip(names, prior.tolist(), strict=True)),
            "arms": {},
        }
        policies = {}
        for stem, label in ARMS.items():
            policy, source = measure.build_policy(
                ROOT / "checkpoints" / "task056-policy-v1" / f"{stem}.pt", config, store, device
            )
            policies[label] = (policy, source)
            features = precompute_features(source, arrays, first)
            predicted = measure.predictions(policy, arrays, first, features)
            error = np.median(np.abs(predicted - targets), 0)
            ratios = {n: float(error[names.index(n)] / thresholds[label][n]) for n in arm_names}
            grasp = predicted[:, names.index("right_grasp")]
            entry["arms"][label] = {
                "median_abs_error": dict(zip(names, error.tolist(), strict=True)),
                "ratio_to_frozen_d1_threshold": ratios,
                "frozen_d1_would_pass": all(v <= 1.0 for v in ratios.values()),
                "predicted_std_per_dimension": dict(
                    zip(names, predicted.std(0).tolist(), strict=True)
                ),
                "d1_grasp_two_sided_count": int((grasp >= 0.0).sum()),
                # Amendment 1's cut, reported beside the frozen one.
                "d1_grasp_amended_count_gt_minus_0_5": int((grasp > -0.5).sum()),
                "step_zero_grasp_min": float(grasp.min()),
                "step_zero_grasp_max": float(grasp.max()),
            }
        report["step_zero"][split] = entry
        if split == "train":
            # §1.1(a)'s train-cohort dz statistics, from committed code (claim audit S6-27).
            dz = base[rows][:, 8]
            report["train_bc_target_right_dz"] = {
                "rows": int(len(rows)),
                "mean": float(dz.mean()),
                "rate_at_plus_0_4": float(np.isclose(dz, 0.4, atol=1e-6).mean()),
                "rate_at_minus_0_4": float(np.isclose(dz, -0.4, atol=1e-6).mean()),
                "rate_abs_at_0_4": float(np.isclose(np.abs(dz), 0.4, atol=1e-6).mean()),
                "median_abs": float(np.median(np.abs(dz))),
            }
            report["d1_pipeline_train_reset_equivalence"] = reset_equivalence(
                arrays, ids, first, policies, measure, precompute_features
            )
            continue

        # D1-pipeline: float-noise floor (single observation, as act() runs it, vs the batched
        # training path) and input sensitivity (what a wrong image or a wrong state does).
        pipeline = {}
        for label, (policy, source) in policies.items():
            batched = measure.predictions(
                policy, arrays, first, precompute_features(source, arrays, first)
            )
            single, swapped, zeroed, later = [], [], [], []
            for k, row in enumerate(first):
                frames = images(arrays, np.array([row]))
                other = first[(k + 1) % len(first)]
                with torch.no_grad():
                    state = policy.normalized_state(arrays.states[[row]], arrays.mask[[row]])
                    state_later = policy.normalized_state(
                        arrays.states[[row + 1]], arrays.mask[[row + 1]]
                    )

                    def run(image_dict, state_tensor, source=source, policy=policy):
                        visual = (
                            source.features(image_dict, inference=True)
                            if source.feature_dim
                            else None
                        )
                        return policy(visual, state_tensor).cpu().numpy()[0]

                    single.append(run(frames, state))
                    swapped.append(run(images(arrays, np.array([other])), state))
                    zeroed.append(run({c: np.zeros_like(v) for c, v in frames.items()}, state))
                    later.append(run(frames, state_later))
            single = np.array(single)
            # The path D1-pipeline actually compares: act() on a RobotState, against the
            # CLIPPED offline prediction (act clips to [-1, 1]; the review of amendment 1 found
            # the step-zero grasp heads sit just below -1, so an unclipped comparison fails a
            # sound pipeline).
            from embodied_jepa.contracts import RobotState

            acted = []
            for row in first:
                state = RobotState(
                    arrays.states[[row]],
                    arrays.mask[[row]],
                    np.asarray(arrays.timestamps[[row]], np.float64),
                    store.state_schema,
                )
                acted.append(
                    policy.act(images(arrays, np.array([row])), state)[list(FREE_ACTION_INDICES)]
                )
            acted = np.array(acted)
            pipeline[label] = {
                "act_vs_clipped_batched_max_abs": float(
                    np.abs(acted - np.clip(batched, -1.0, 1.0)).max()
                ),
                "act_vs_unclipped_batched_max_abs": float(np.abs(acted - batched).max()),
                "float_noise_max_abs_single_vs_batched": float(np.abs(single - batched).max()),
                "swapped_reset_image_max_abs_change": float(
                    np.abs(np.array(swapped) - single).max(axis=1).min()
                ),
                "swapped_reset_image_median_of_per_frame_max_change": float(
                    np.median(np.abs(np.array(swapped) - single).max(axis=1))
                ),
                "zero_image_min_of_per_frame_max_change": float(
                    np.abs(np.array(zeroed) - single).max(axis=1).min()
                ),
                "next_frame_state_min_of_per_frame_max_change": float(
                    np.abs(np.array(later) - single).max(axis=1).min()
                ),
            }
        report["d1_pipeline_calibration"] = pipeline

        # D2: open-loop first departure on every command row of every surviving val root.
        spans = []
        for e in ids:
            index = arrays.episode_ids.index(e)
            start, length = int(arrays.offsets[index]), int(arrays.lengths[index])
            spans.append(np.arange(start, start + length - 1))
        every = np.concatenate(spans)
        target_rows = base[every][:, list(FREE_ACTION_INDICES)]
        mean_blind = np.asarray(blind_tables(surviving_train_roots())[0])
        departure_report = {}
        for label, (policy, source) in policies.items():
            tau = 3.0 * float(np.median([table_a[label][n] for n in arm_names]))
            predicted = measure.predictions(
                policy, arrays, every, precompute_features(source, arrays, every)
            )
            arm_entry = {"tau": tau}
            for kind, values in (
                ("arm", predicted),
                ("prior_only_mean", np.broadcast_to(mean_blind, predicted.shape)),
            ):
                dep = np.median(np.abs(values[:, :6] - target_rows[:, :6]), axis=1)
                for k in (1, pd.DEPARTURE_PERSISTENCE, 10, 20):
                    firsts, offset = [], 0
                    for span in spans:
                        found = pd.first_departure_step(
                            [float(v) for v in dep[offset : offset + len(span)]], tau, k
                        )
                        firsts.append(len(span) if found is None else found)
                        offset += len(span)
                    arm_entry[f"{kind}_persistence_{k}_median_first_departure"] = float(
                        np.median(firsts)
                    )
                arm_entry[f"{kind}_rate_above_tau"] = float((dep > tau).mean())
            departure_report[label] = arm_entry
        report["d2_departure_calibration"] = departure_report
    report["elapsed_seconds"] = time.perf_counter() - started
    _write(output, report)
    return report


def reset_equivalence(arrays, ids, first, policies, measure, precompute_features):
    """A sound-pipeline demonstration on TRAIN roots (disjoint from D1-pipeline's val roots).

    Resets the simulator to each provenance root's recorded reset exactly as the runner will,
    executes nothing, and compares the live observation and every arm's act() with the stored
    frame and the clipped offline prediction. This is what "a sound pipeline passes D1" rests on.
    """
    from embodied_jepa.policy import FREE_ACTION_INDICES

    c = collector()
    plan = {r["seed"]: r for r in c.make_plan(c.FROZEN_SEEDS)["roots"]}
    by_seed = {int(e.split("-")[1]): int(row) for e, row in zip(ids, first, strict=True)}
    out = {"roots": list(PROVENANCE_ROOTS), "per_root": {}}
    worst = {label: 0.0 for label in policies}
    for seed in PROVENANCE_ROOTS:
        row = by_seed[seed]
        robot = _robot()
        try:
            robot.observe()
            robot.reset(
                seed=seed, object_xy=plan[seed]["object_xy"], plate_xy=plan[seed]["plate_xy"]
            )
            obs = robot.observe()
        finally:
            robot.sim.close()
        entry = {
            "image_identical": bool(
                np.array_equal(obs.images["onboard_rgb"][0], arrays.frames["onboard_rgb"][row])
            ),
            "state_max_abs": float(np.abs(obs.robot_state[0] - arrays.states[row]).max()),
            "mask_identical": bool(np.array_equal(obs.state_mask[0], arrays.mask[row])),
            "act_vs_clipped_offline_max_abs": {},
        }
        for label, (policy, source) in policies.items():
            one = np.array([row])
            offline = measure.predictions(
                policy, arrays, one, precompute_features(source, arrays, one)
            )[0]
            live = policy.act(obs.images, obs.state)[list(FREE_ACTION_INDICES)]
            diff = float(np.abs(live - np.clip(offline, -1.0, 1.0)).max())
            entry["act_vs_clipped_offline_max_abs"][label] = diff
            worst[label] = max(worst[label], diff)
        out["per_root"][str(seed)] = entry
    out["all_images_identical"] = all(e["image_identical"] for e in out["per_root"].values())
    out["max_state_abs"] = max(e["state_max_abs"] for e in out["per_root"].values())
    out["worst_act_vs_clipped_offline"] = worst
    return out


def _write(output, report):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("provenance", "closed-loop", "offline"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.part == "provenance":
        report = provenance(args.output, args.workers)
        print(json.dumps(report["replays"], indent=1))
    elif args.part == "closed-loop":
        report = closed_loop(args.output, args.workers)
        print(
            json.dumps(
                {k: (v["grasp_resets"], v["full_successes"]) for k, v in report["runs"].items()},
                indent=1,
            )
        )
    else:
        report = offline(args.output, args.workers, args.device)
        print(json.dumps(report["step_zero"]["val"]["arms"], indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
