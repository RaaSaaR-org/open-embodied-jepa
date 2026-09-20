"""Bounded replay/open-loop forecast audit; no fitting or TEST image decoding."""

from __future__ import annotations

import time

ENTRY = time.time(), time.monotonic()

# ruff: noqa: E402
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOTS = (119, 300, 550)
KINDS = ("winner", "demonstration", "hold")
TRACE_SHA = "921eacf54ca7cb131ef5fe1abb7f99e32af11bb5423dc56fae459664286df4c2"
CHECKPOINT_SHA = "0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5"
DATA_SHA = "6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331"
PROTOCOL = "docs/experiments/apple_control_forecast_audit_v1.md"


def elapsed(entry=ENTRY):
    return max(0.0, time.time() - entry[0], time.monotonic() - entry[1])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_array(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_array) + "\n"
    )
    temp.replace(path)


def trace_path(original):
    return original / "attempts/43000-learned/trace.jsonl"


def load_trace(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    results = [row for row in rows if row.get("event") == "result" and row.get("executed") is True]
    if len(results) != 574 or [r["step"] for r in results] != list(range(574)):
        raise ValueError("original trace must contain exactly574 consecutive accepted commands")
    return results


def identities(repository, original):
    plan = json.loads((original / "resolved_plan.json").read_text())
    paths = {
        "trace": trace_path(original),
        "resolved_plan": original / "resolved_plan.json",
        "waypoints": original / "waypoints.npz",
        "calibration": original / "calibration.json",
        "checkpoint": Path(plan["checkpoint"]),
        "dataset_manifest": Path(plan["dataset"]) / "meta/jepa_manifest.json",
        "script": Path(__file__),
        "protocol": repository / PROTOCOL,
    }
    paths.update({f"runtime/{name}": original / "source" / name for name in plan["source_hashes"]})
    paths["action"] = original / "source/configs/g1_sim_action.json"
    paths["asset_manifest"] = original / "source/assets/manifest.json"
    result = {name: digest(path) for name, path in paths.items()}
    result["repository_revision"] = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        result["trace"] != TRACE_SHA
        or result["checkpoint"] != CHECKPOINT_SHA
        or result["dataset_manifest"] != DATA_SHA
    ):
        raise ValueError("original trace/checkpoint/dataset differs from preregistration")
    return result


def planned_records():
    return [
        dict(root_step=root, kind=kind, status="not_started", executed_steps=0)
        for root in ROOTS
        for kind in KINDS
    ]


def assert_trace_match(expected, actual, *, acknowledged=False):
    import numpy as np

    for key in (
        "step",
        "goal_index",
        "goal_source_id",
        "selected_origin",
        "selected_round",
        "goal_dwell_observed",
        "goal_image_sha256",
        "waypoint_advanced",
    ):
        if expected[key] != actual[key]:
            raise ValueError(f"replay {key} mismatch at{expected['step']}")
    for key in (
        "sampled_action",
        "projected_action",
        "observed_distance",
        "selected_cost",
        "goal_threshold",
    ):
        if not np.allclose(expected[key], actual[key], atol=1e-6, rtol=1e-5):
            raise ValueError(f"replay {key} mismatch at{expected['step']}")
    for old, new in zip(expected["rounds"], actual["rounds"], strict=True):
        for key in ("selected_index", "feasible_candidates", "score_permutation"):
            if old[key] != new[key]:
                raise ValueError(f"replay candidate {key} mismatch")
        for a, b in zip(old["candidate_costs"], new["candidate_costs"], strict=True):
            if (a is None) != (b is None) or (
                a is not None and not np.isclose(a, b, atol=1e-6, rtol=1e-5)
            ):
                raise ValueError("replay candidate ranking/cost mismatch")
    for key in ("observation_timestamp",) + (("execution_timestamp",) if acknowledged else ()):
        if not np.isclose(expected[key], actual[key], atol=1e-9, rtol=0):
            raise ValueError(f"replay {key} mismatch")
    if acknowledged and not np.allclose(
        expected["applied_action"], actual["applied_action"], atol=1e-6, rtol=0
    ):
        raise ValueError("replay actual-applied command mismatch")


def assert_score_match(expected, actual):
    import numpy as np

    if set(expected) != set(actual):
        raise ValueError("replay score fields changed")
    for key, value in expected.items():
        observed = actual[key]
        if value is None or isinstance(value, (bool, str)):
            if type(value) is not type(observed) or observed != value:
                raise ValueError(f"replay ordered score mismatch:{key}")
        elif not isinstance(observed, (int, float)) or not np.isclose(
            value, observed, atol=1e-6, rtol=1e-5
        ):
            raise ValueError(f"replay scoring value mismatch:{key}")


def recover_candidates(trace, captures, proposal_count):
    """Recover scored sequences without consuming the controller RNG."""
    round_index = trace["selected_round"]
    winner_index = trace["rounds"][round_index]["selected_index"]
    options = []
    for iteration, part in enumerate(trace["rounds"]):
        for index in range(2, 2 + proposal_count):
            value = part["candidate_costs"][index]
            if value is not None:
                options.append((value, iteration, index))
    if not options:
        raise ValueError("no scored feasible demonstration; no substitute allowed")
    _, demo_round, demo_index = min(options)
    records = {}
    for kind, iteration, index in [
        ("winner", round_index, winner_index),
        ("demonstration", demo_round, demo_index),
        ("hold", 0, 0),
    ]:
        captured = captures[iteration]
        if not captured["feasible"][index]:
            raise ValueError(f"{kind} candidate infeasible")
        records[kind] = dict(
            round=iteration,
            index=index,
            cost=trace["rounds"][iteration]["candidate_costs"][index],
            requested=captured["requested"][index].copy(),
            scored=captured["actions"][index].copy(),
        )
    return records


def capture_robot(robot, scorer):
    data = robot.mj.MjData(robot.model)
    robot.mj.mj_copyData(data, robot.model, robot.sim.data)
    return dict(
        data=data,
        targets=robot.sim.targets.copy(),
        grasp=robot._grasp.copy(),
        stopped_reason=robot.sim.stopped_reason,
        scorer={k: copy.deepcopy(v) for k, v in vars(scorer).items() if k != "robot"},
    )


def restore_robot(robot, scorer, saved):
    robot.mj.mj_copyData(robot.sim.data, robot.model, saved["data"])
    robot.sim.targets[:] = saved["targets"]
    robot._grasp[:] = saved["grasp"]
    robot.sim.stopped_reason = saved["stopped_reason"]
    robot._observation = None
    for key, value in saved["scorer"].items():
        setattr(scorer, key, copy.deepcopy(value))
    return robot.observe()


def capture_controller(controller):
    excluded = {"model", "waypoints", "config", "progress_distance", "_goal_latents"}
    # Cached model-owned immutable latents retain their owner identity.
    return (
        {
            key: copy.deepcopy(value)
            for key, value in vars(controller).items()
            if key not in excluded
        },
        controller._goal_latents.copy(),
    )


def restore_controller(controller, saved):
    state, latents = saved
    for key, value in state.items():
        setattr(controller, key, copy.deepcopy(value))
    controller._goal_latents = latents.copy()


def save_arrays(path, **arrays):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, **arrays)
    temp.replace(path)


def persist_observation(path, observation, **arrays):
    save_arrays(
        path,
        rgb=observation.images["onboard_rgb"][0],
        state=observation.robot_state[0],
        mask=observation.state_mask[0],
        timestamp=observation.timestamps[0],
        **arrays,
    )


def same_observation(first, second):
    import numpy as np

    return all(
        np.array_equal(a, b)
        for a, b in [
            (first.images["onboard_rgb"], second.images["onboard_rgb"]),
            (first.robot_state, second.robot_state),
            (first.state_mask, second.state_mask),
            (first.timestamps, second.timestamps),
        ]
    )


def forecast_metrics(model, prediction, observation, goal, horizon):
    import torch

    with torch.no_grad():
        values = prediction.values[0, 0, horizon - 1]
        raw = values * model.sensor_scale + model.sensor_mean
        pixels, _ = model.pixels(observation.images)
        n = model.visual_dimension
        state = torch.from_numpy(observation.robot_state[0].copy()).to(raw.device)
        half = len(state) // 2
        costs = model.distance(prediction, goal)
        return dict(
            predicted_goal_distance=float(costs[0, 0, horizon - 1]),
            raw_rgb_grid_mse=float((raw[:n] - pixels.flatten()).square().mean().item()),
            qpos_mse_rad2=float((raw[n : n + half] - state[:half]).square().mean().item()),
            qvel_mse_rad2_s2=float((raw[n + half :] - state[half:]).square().mean().item()),
        )


def reforecast_actual(model, root_latent, applied):
    import numpy as np

    values = np.asarray(applied, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 14 or len(values) == 0:
        raise ValueError("actual forecast requires a nonempty accepted [T,14] prefix")
    return model.predict(root_latent, values[None, None])


def worker(args):
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    import numpy as np
    import torch

    torch.set_num_threads(4)
    original = args.original.resolve()
    runtime = original / "source"
    sys.path.insert(0, str(runtime / "src"))
    helper_spec = importlib.util.spec_from_file_location(
        "frozen_evaluate_apple", runtime / "scripts/evaluate_apple.py"
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation
    from embodied_jepa.task import AppleToPlateTask
    from embodied_jepa.waypoint_planning import WaypointConfig, WaypointController

    plan = json.loads((args.output / "plan.json").read_text())
    report = dict(
        status="running", records=planned_records(), replayed_steps=0, test_images_decoded=False
    )
    robot = None

    def persist():
        report.update(wall_seconds=elapsed((args.wall, args.mono)), cpu_seconds=time.process_time())
        write(args.output / "report.json", report)

    def check():
        if elapsed((args.wall, args.mono)) >= 165 or time.process_time() >= 115:
            raise TimeoutError("audit execution budget exhausted; finalization reserved")

    persist()
    try:
        if identities(args.repository, original) != plan["identities"]:
            raise ValueError("audit frozen inputs changed")
        control = json.loads((original / "resolved_plan.json").read_text())
        helper.verify_plan_inputs(control, source_root=runtime)
        trace = load_trace(trace_path(original))
        store, model = helper.checkpoint_model(control["dataset"], control["checkpoint"])
        if model.backend != "sensor_wm":
            raise ValueError("diagnostic requires frozen sensor_wm")
        waypoints = helper.load_waypoints(original, control)
        trial = next(r for r in control["attempts"] if r["attempt_id"] == "43000-learned")
        controller = WaypointController(
            model,
            waypoints,
            progress_distance=model.observed_distance,
            config=WaypointConfig(
                **(control["controller"] | {"seed": 43000, "ablation": "learned"})
            ),
        )
        robot = G1Embodiment(
            MuJoCoSimulation(object_kind="apple", container_kind="plate", width=96, height=96)
        )
        if robot.state_schema != store.state_schema or helper.json_hash(
            robot.manifest
        ) != helper.json_hash(store.manifest["action_manifest"]):
            raise ValueError("frozen embodiment differs from dataset state/action contracts")
        robot.reset(**trial["reset"])
        scorer = AppleToPlateTask(robot)
        for step in range(max(ROOTS) + 1):
            check()
            planning_start = (time.time(), time.monotonic())
            observation = robot.observe()
            captures = []
            before_controller = capture_controller(controller) if step in ROOTS else None

            def projector(requested, captures=captures):
                result = robot.project_candidates(requested)
                captures.append(
                    dict(
                        requested=requested[0].copy(),
                        actions=result.actions[0].copy(),
                        feasible=result.feasible[0].copy(),
                    )
                )
                return result

            decision = controller.step(observation, projector)
            if decision.action is None:
                raise ValueError("replay terminated before fixed roots")
            assert_trace_match(trace[step], decision.trace)
            planning_seconds = elapsed(planning_start)
            report["replay_max_plan_seconds"] = max(
                report.get("replay_max_plan_seconds", 0), planning_seconds
            )
            if planning_seconds > control["control_timeout_seconds"]:
                raise TimeoutError(
                    "replay controller exceeded original deadline including capture overhead"
                )
            if step in ROOTS:
                saved = capture_robot(robot, scorer)
                after_controller = capture_controller(controller)
                wp = waypoints[controller.goal_index]
                candidates = recover_candidates(
                    decision.trace,
                    captures,
                    len(wp.action_proposals) if wp.action_proposals is not None else 0,
                )
                if not np.allclose(
                    candidates["winner"]["scored"][0], decision.action, atol=1e-6, rtol=0
                ):
                    raise ValueError("recovered winner differs from replay command")
                rootdir = args.output / f"root-{step}"
                persist_observation(rootdir / "initial.npz", observation)
                spec = robot.mj.mjtState.mjSTATE_INTEGRATION
                integration = np.empty(robot.mj.mj_stateSize(robot.model, spec))
                robot.mj.mj_getState(robot.model, robot.sim.data, integration, spec)
                save_arrays(
                    rootdir / "simulator_state.npz",
                    integration=integration,
                    targets=saved["targets"],
                    grasp=saved["grasp"],
                )
                # Save controller state without serializing model objects or owner tokens.
                for label, snapshot in [
                    ("before_plan", before_controller),
                    ("after_plan", after_controller),
                ]:
                    values, cache = snapshot
                    arrays = {
                        f"goal_{i}": v.values.detach().cpu().numpy() for i, v in cache.items()
                    }
                    if values["warm"] is not None:
                        arrays["warm"] = values["warm"]
                    arrays["last_grasps"] = values["last_grasps"]
                    save_arrays(rootdir / f"controller_{label}.npz", **arrays)
                    write(
                        rootdir / f"controller_{label}.json",
                        {
                            "rng": values["rng"].bit_generator.state,
                            **{
                                key: values[key]
                                for key in (
                                    "goal_index",
                                    "dwell",
                                    "steps",
                                    "last_timestamp",
                                    "last_execution_timestamp",
                                    "termination_reason",
                                )
                            },
                            "pending": None
                            if values["pending"] is None
                            else {
                                "action": values["pending"][0].tolist(),
                                "sequence": values["pending"][1].tolist(),
                                "trace": values["pending"][2],
                            },
                            "scorer": saved["scorer"],
                        },
                    )
                latent = model.encode(observation.images, observation.state)
                goal = model.encode_goal(wp.images)
                for kind in KINDS:
                    check()
                    record = next(
                        r for r in report["records"] if r["root_step"] == step and r["kind"] == kind
                    )
                    record.update(
                        status="running",
                        goal_index=controller.goal_index,
                        goal_source_id=wp.source_id,
                        initial_goal_distance=float(
                            model.observed_distance(observation.images, wp.images)[0]
                        ),
                    )
                    persist()
                    candidate = candidates[kind]
                    directory = rootdir / kind
                    restored = restore_robot(robot, scorer, saved)
                    restore_controller(controller, after_controller)
                    if not same_observation(observation, restored):
                        raise ValueError("branch initial sensor mismatch")
                    planned = model.predict(latent, candidate["scored"][None, None])
                    if not np.isclose(
                        model.distance(planned, goal)[0, 0, -1],
                        candidate["cost"],
                        atol=1e-6,
                        rtol=1e-5,
                    ):
                        raise ValueError(
                            "recovered sequence forecast disagrees with original score"
                        )
                    record.update(
                        candidate_round=candidate["round"],
                        candidate_index=candidate["index"],
                        scored_cost=candidate["cost"],
                        metrics={
                            str(h): {"status": "unavailable", "reason": "not_evaluated"}
                            for h in (1, 16)
                        },
                    )
                    save_arrays(
                        directory / "scored.npz",
                        requested=candidate["requested"],
                        projected=candidate["scored"],
                        predicted=planned.values.detach().cpu().numpy(),
                    )
                    applied = []
                    measured_endpoints = {}
                    for index, command in enumerate(candidate["scored"]):
                        check()
                        write(
                            directory / "command.json",
                            dict(index=index + 1, status="pending", requested=command.tolist()),
                        )
                        try:
                            result = robot.execute(command)
                            if result.applied_action is None:
                                record.update(
                                    status="command_rejected",
                                    reason=result.reason,
                                    rejected_step=index + 1,
                                )
                                write(
                                    directory / "command.json",
                                    dict(index=index + 1, status="rejected", reason=result.reason),
                                )
                                break
                            measured = robot.observe()
                            persist_observation(
                                directory / f"frame-{index + 1:02d}.npz",
                                measured,
                                applied=result.applied_action,
                                requested=command,
                            )
                            applied.append(result.applied_action.copy())
                            record["executed_steps"] = len(applied)
                            score = scorer.evaluate()
                            write(directory / f"score-{index + 1:02d}.json", score)
                            if index + 1 in (1, 16):
                                measured_endpoints[index + 1] = measured
                        except Exception as error:
                            record.update(
                                status="runtime_error",
                                error=f"{type(error).__name__}: {error}",
                                possible_unobserved_execution=True,
                            )
                            break
                    if record["status"] == "running":
                        record["status"] = "completed"
                    for horizon in (1, 16):
                        if horizon not in measured_endpoints:
                            record["metrics"][str(horizon)] = {
                                "status": "unavailable",
                                "reason": "horizon_not_observed",
                            }
                    if applied:
                        actual = np.asarray(applied, np.float32)
                        recomputed = reforecast_actual(model, latent, actual)
                        save_arrays(
                            directory / "actual.npz",
                            applied=actual,
                            predicted=recomputed.values.detach().cpu().numpy(),
                        )
                        delta = np.abs(actual - candidate["scored"][: len(actual)])
                        record.update(
                            applied_diff_steps=int((delta > 1e-6).any(1).sum()),
                            applied_diff_max=float(delta.max()),
                            applied_diff_mean=float(delta.mean()),
                        )
                        for horizon, measured in measured_endpoints.items():
                            record["metrics"][str(horizon)] = {
                                "status": "measured",
                                "observed_goal_distance": float(
                                    model.observed_distance(measured.images, wp.images)[0]
                                ),
                                "original_scored_actions": forecast_metrics(
                                    model, planned, measured, goal, horizon
                                ),
                                "actual_applied_actions": forecast_metrics(
                                    model, recomputed, measured, goal, horizon
                                ),
                            }
                    persist()
                restore_robot(robot, scorer, saved)
                restore_controller(controller, after_controller)
            if step < max(ROOTS):
                check()
                result = robot.execute(decision.action)
                if result.applied_action is None:
                    raise ValueError(f"replay command rejected:{result.reason}")
                acknowledged = controller.acknowledge(result)
                assert_trace_match(trace[step], acknowledged, acknowledged=True)
                score = scorer.evaluate()
                assert_score_match(trace[step]["score"], score)
                report["replayed_steps"] = step + 1
            persist()
        helper.verify_plan_inputs(control, source_root=runtime)
        store.verify()
        if identities(args.repository, original) != plan["identities"]:
            raise ValueError("audit inputs changed during run")
        report.update(status="completed", integrity_verified_after=True)
        if any(r["status"] not in ("completed", "command_rejected") for r in report["records"]):
            report["status"] = "incomplete"
    except Exception as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        if robot is not None:
            robot.close()
        persist()
    return 0 if report["status"] == "completed" else 2


def durable_prefix(directory):
    import numpy as np

    count = 0
    issues = []
    for path in sorted(directory.glob("frame-*.npz")):
        try:
            index = int(path.stem.split("-")[-1])
            if index != count + 1:
                raise ValueError("gapped frame sequence")
            with np.load(path, allow_pickle=False) as values:
                for key in ("applied", "requested", "rgb", "state", "mask", "timestamp"):
                    if key not in values:
                        raise ValueError(f"missing {key}")
                if values["applied"].shape != (14,) or values["requested"].shape != (14,):
                    raise ValueError("invalid action shape")
                if values["state"].ndim != 1 or values["mask"].shape != values["state"].shape:
                    raise ValueError("invalid sensor shape")
                if any(
                    not np.isfinite(values[key]).all()
                    for key in ("applied", "requested", "state", "timestamp")
                ):
                    raise ValueError("nonfinite journal")
            count += 1
        except Exception as error:
            issues.append(f"{path.name}: {error}")
            break
    uncertain = bool(issues)
    command = directory / "command.json"
    if command.exists():
        try:
            current = json.loads(command.read_text())
            if type(current.get("index")) is not int or current.get("status") not in (
                "pending",
                "rejected",
            ):
                raise ValueError("invalid pending-command record")
            uncertain |= current["status"] == "pending" and current["index"] > count
        except Exception as error:
            issues.append(f"command.json: {error}")
            uncertain = True
    return count, uncertain, issues


def finalize(output, returncode, timed_out):
    path = output / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    records = {f"{r['root_step']}-{r['kind']}": r for r in report.get("records", [])}
    final = []
    for declared in planned_records():
        row = records.get(f"{declared['root_step']}-{declared['kind']}", declared)
        if row["status"] == "running":
            directory = output / f"root-{row['root_step']}" / row["kind"]
            confirmed, uncertain, issues = durable_prefix(directory)
            row.update(
                status="interrupted",
                executed_steps=confirmed,
                durable_prefix_frames=confirmed,
                possible_unobserved_execution=uncertain,
                journal_issues=issues,
            )
            row.setdefault(
                "metrics",
                {
                    str(h): {"status": "unavailable", "reason": "not_computed_before_interruption"}
                    for h in (1, 16)
                },
            )
        elif row["status"] == "not_started":
            row["status"] = "not_started_budget" if timed_out else "not_started_failure"
        final.append(row)
    complete = (
        returncode == 0
        and not timed_out
        and report.get("status") == "completed"
        and report.get("integrity_verified_after") is True
    )
    report.update(
        records=final,
        status="completed" if complete else "incomplete",
        returncode=returncode,
        supervisor_timeout=timed_out,
        supervisor_wall_seconds=elapsed(),
    )
    write(path, report)
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wall", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mono", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.repository = args.repository.resolve()
    args.original = args.original.resolve()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite forecast audit")
    args.output.mkdir(parents=True)
    write(
        args.output / "plan.json",
        dict(
            roots=list(ROOTS),
            records=planned_records(),
            max_wall_seconds=180,
            max_cpu_seconds=120,
            worker_kill_seconds=175,
            no_test_decode=True,
        ),
    )
    try:
        identity = identities(args.repository, args.original)
        load_trace(trace_path(args.original))
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name) for name in ("torch", "mujoco", "numpy")
            },
        }
        write(
            args.output / "plan.json",
            dict(
                roots=list(ROOTS),
                records=planned_records(),
                identities=identity,
                environment=environment,
                max_wall_seconds=180,
                max_cpu_seconds=120,
                worker_kill_seconds=175,
                replay_tolerance={"atol": 1e-6, "rtol": 1e-5},
                no_test_decode=True,
            ),
        )
    except Exception as error:
        write(
            args.output / "report.json",
            dict(
                status="incomplete",
                records=planned_records(),
                error=f"{type(error).__name__}: {error}",
            ),
        )
        return finalize(args.output, 2, False)
    if elapsed() >= 175:
        return finalize(args.output, 2, True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repository",
        str(args.repository),
        "--original",
        str(args.original),
        "--output",
        str(args.output),
        "--wall",
        str(ENTRY[0]),
        "--mono",
        str(ENTRY[1]),
    ]
    with (args.output / "worker.log").open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=args.repository,
            env=os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        timeout = False
        while child.poll() is None:
            if elapsed() >= 175:
                child.kill()
                timeout = True
                break
            time.sleep(0.1)
        returncode = child.wait()
    return finalize(args.output, returncode, timeout)


if __name__ == "__main__":
    raise SystemExit(main())
