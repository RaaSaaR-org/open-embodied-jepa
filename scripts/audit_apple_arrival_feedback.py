"""Paired development-state arrival-feedback audit; no training or TEST decoding."""

from __future__ import annotations

import time

ENTRY = (time.time(), time.monotonic())
# ruff: noqa: E402
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

ROOT_STEP = 60
KINDS = ("original", "arrival_interrupt")
TRACE_SHA = "f59e7eacfd22cfb1ff51cd0e85a127e6e91525dd5fe39e8303f5358506c2f90c"
CHECKPOINT_SHA = "0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5"
DATA_SHA = "6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331"
PROTOCOL = "docs/experiments/apple_arrival_feedback_v1.md"


def elapsed(entry=ENTRY):
    return max(0.0, time.time() - entry[0], time.monotonic() - entry[1])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def planned():
    return [
        dict(
            kind=k,
            status="not_started",
            attempted_commands=0,
            executed_steps=0,
            advanced_past_goal2=None,
            unavailable_reason="not_started",
            command_slots=[dict(index=i, status="not_started") for i in range(16)],
        )
        for k in KINDS
    ]


def load_trace(original):
    rows = [
        json.loads(line)
        for line in (original / "attempts/43000-learned/trace.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r.get("event") == "result" and r.get("executed") is True]
    if len(rows) != 1000 or [r["step"] for r in rows] != list(range(1000)):
        raise ValueError("require sealed 1000-command learned trace")
    root = rows[ROOT_STEP]
    if (
        root["goal_index"] != 2
        or root["goal_dwell_observed"] != 1
        or root["commitment_offset"] != 2
    ):
        raise ValueError("preregistered arrival root identity changed")
    return rows


def identities(args):
    control = json.loads((args.original / "resolved_plan.json").read_text())
    paths = dict(
        trace=args.original / "attempts/43000-learned/trace.jsonl",
        report=args.original / "attempts/43000-learned/report.json",
        resolved=args.original / "resolved_plan.json",
        waypoints=args.original / "waypoints.npz",
        calibration=args.original / "calibration.json",
        checkpoint=Path(control["checkpoint"]),
        dataset=Path(control["dataset"]) / "meta/jepa_manifest.json",
        script=Path(__file__),
        protocol=args.repository / PROTOCOL,
        helpers=args.repository / "scripts/audit_apple_control_forecasts.py",
    )
    paths.update({f"runtime/{k}": args.original / "source" / k for k in control["source_hashes"]})
    paths.update(
        action=args.original / "source/configs/g1_sim_action.json",
        assets=args.original / "source/assets/manifest.json",
    )
    result = {k: digest(v) for k, v in paths.items()}
    if (
        result["trace"] != TRACE_SHA
        or result["checkpoint"] != CHECKPOINT_SHA
        or result["dataset"] != DATA_SHA
    ):
        raise ValueError("frozen experiment inputs changed")
    result["revision"] = subprocess.check_output(
        ["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True
    ).strip()
    return result


def arrival_interrupt(controller, observation):
    """Measure only images; normal step remains solely responsible for dwell/RNG."""
    if controller.termination_reason is not None or controller.pending is not None:
        return None
    waypoint = controller.waypoints[controller.goal_index]
    distance = controller._distance(observation.images, waypoint)
    if distance <= waypoint.threshold and controller._commitment is not None:
        cache = controller._commitment
        event = dict(
            reason="measured_image_arrival",
            distance=distance,
            threshold=waypoint.threshold,
            plan_step=cache["trace"]["plan_step"],
            commitment_offset=cache["offset"],
        )
        controller._commitment = None
        return event
    return None


def assert_match(helper, old, new, *, acknowledged=False):
    helper.assert_trace_match(old, new, acknowledged=acknowledged)
    for key in (
        "decision_kind",
        "plan_step",
        "commitment_offset",
        "commitment_remaining",
        "commitment_abort_reason",
        "commitment_clear_reason",
        "commitment_remaining_after_ack",
    ):
        if (
            key
            in (
                "commitment_abort_reason",
                "commitment_clear_reason",
                "commitment_remaining_after_ack",
            )
            and not acknowledged
        ):
            continue
        if old.get(key) != new.get(key):
            raise ValueError(f"replay commitment mismatch: {key}")


def recover_trace(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    results = [r for r in rows if r.get("event") == "result"]
    if [r["index"] for r in results] != list(range(len(results))):
        raise ValueError("noncontiguous durable result journal")
    pending = [r["index"] for r in rows if r.get("event") == "pending"]
    uncertain = bool(pending and pending[-1] >= len(results))
    return results, uncertain


def worker(args):
    resource.setrlimit(resource.RLIMIT_CPU, (40, 40))
    import torch

    torch.set_num_threads(4)
    runtime = args.original / "source"
    sys.path.insert(0, str(runtime / "src"))
    helpers = load_module(
        "forecast_helpers", args.repository / "scripts/audit_apple_control_forecasts.py"
    )
    evaluator = load_module("frozen_evaluator", runtime / "scripts/evaluate_apple.py")
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation
    from embodied_jepa.task import AppleToPlateTask
    from embodied_jepa.waypoint_planning import WaypointConfig, WaypointController

    report = dict(status="running", records=planned(), replayed_steps=0, test_images_decoded=False)
    robot = None

    def persist():
        report.update(wall_seconds=elapsed((args.wall, args.mono)), cpu_seconds=time.process_time())
        write(args.output / "report.json", report)

    def check():
        if elapsed((args.wall, args.mono)) >= 53 or time.process_time() >= 37:
            raise TimeoutError("execution budget reached; preserve finalization reserve")

    def append(path, row):
        with path.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    try:
        persist()
        check()
        expected = json.loads((args.output / "plan.json").read_text())["identities"]
        if identities(args) != expected:
            raise ValueError("identity mismatch before execution")
        control = json.loads((args.original / "resolved_plan.json").read_text())
        evaluator.verify_plan_inputs(control, source_root=runtime)
        trace = load_trace(args.original)
        store, model = evaluator.checkpoint_model(control["dataset"], control["checkpoint"])
        waypoints = evaluator.load_waypoints(args.original, control)
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
        if robot.state_schema != store.state_schema or evaluator.json_hash(
            robot.manifest
        ) != evaluator.json_hash(store.manifest["action_manifest"]):
            raise ValueError("embodiment/data contract mismatch")
        trial = next(t for t in control["attempts"] if t["attempt_id"] == "43000-learned")
        robot.reset(**trial["reset"])
        scorer = AppleToPlateTask(robot)
        for step in range(ROOT_STEP):
            check()
            clock = (time.time(), time.monotonic())
            observation = robot.observe()
            decision = controller.step(observation, robot.project_candidates)
            assert_match(helpers, trace[step], decision.trace)
            if elapsed(clock) > control["control_timeout_seconds"]:
                raise TimeoutError("replay control deadline")
            check()
            result = robot.execute(decision.action)
            ack = controller.acknowledge(result)
            assert_match(helpers, trace[step], ack, acknowledged=True)
            score = scorer.evaluate()
            helpers.assert_score_match(trace[step]["score"], score)
            report["replayed_steps"] = step + 1
            persist()
        initial = robot.observe()
        saved_robot = helpers.capture_robot(robot, scorer)
        saved_controller = helpers.capture_controller(controller)
        helpers.persist_observation(args.output / "root-60.npz", initial)
        # Verify the root decision without executing it, then restore all mutable state.
        decision = controller.step(initial, robot.project_candidates)
        assert_match(helpers, trace[ROOT_STEP], decision.trace)
        helpers.restore_controller(controller, saved_controller)
        for row in report["records"]:
            check()
            kind = row["kind"]
            folder = args.output / kind
            folder.mkdir()
            observation = helpers.restore_robot(robot, scorer, saved_robot)
            if not helpers.same_observation(initial, observation):
                raise ValueError("root restore sensor mismatch")
            helpers.restore_controller(controller, saved_controller)
            row.update(status="running", advanced_past_goal2=False, unavailable_reason=None)
            persist()
            try:
                for index in range(16):
                    check()
                    clock = (time.time(), time.monotonic())
                    observation = robot.observe()
                    interrupt = (
                        arrival_interrupt(controller, observation)
                        if kind == "arrival_interrupt"
                        else None
                    )
                    previous_goal_index = controller.goal_index
                    decision = controller.step(observation, robot.project_candidates)
                    if controller.goal_index > 2:
                        row["advanced_past_goal2"] = True
                        row.setdefault("first_advance_command", index)
                    if previous_goal_index == 2 and controller.goal_index > 2:
                        # Normal step has already reset dwell for the new goal;
                        # advancement certifies the old goal's required dwell.
                        row["max_goal2_dwell"] = waypoints[2].dwell_observations
                        row["goal2_completed_dwell_observations"] = waypoints[2].dwell_observations
                    if decision.action is None:
                        row["termination_reason"] = decision.termination_reason
                        break
                    if elapsed(clock) > control["control_timeout_seconds"]:
                        raise TimeoutError("sibling control deadline")
                    if kind == "original":
                        assert_match(helpers, trace[ROOT_STEP + index], decision.trace)
                    check()
                    append(
                        folder / "trace.jsonl",
                        dict(
                            event="pending",
                            index=index,
                            trace=decision.trace,
                            arrival_interrupt=interrupt,
                        ),
                    )
                    row["attempted_commands"] = index + 1
                    row["command_slots"][index]["status"] = "pending"
                    result = robot.execute(decision.action)
                    ack = controller.acknowledge(result)
                    accepted = result.applied_action is not None
                    # Ack is durable before scorer so a scoring failure cannot erase execution.
                    append(
                        folder / "trace.jsonl",
                        dict(
                            event="result",
                            index=index,
                            executed=accepted,
                            trace=ack,
                            arrival_interrupt=interrupt,
                        ),
                    )
                    row["executed_steps"] += int(accepted)
                    row["command_slots"][index]["status"] = "accepted" if accepted else "rejected"
                    score = scorer.evaluate()
                    row["score"] = score
                    append(folder / "scores.jsonl", dict(index=index, score=score))
                    if kind == "original":
                        assert_match(helpers, trace[ROOT_STEP + index], ack, acknowledged=True)
                        helpers.assert_score_match(trace[ROOT_STEP + index]["score"], score)
                    row["last_goal_index"] = controller.goal_index
                    row["last_distance"] = decision.trace["observed_distance"]
                    if decision.trace["goal_index"] == 2:
                        row["max_goal2_dwell"] = max(
                            row.get("max_goal2_dwell", 0), decision.trace["goal_dwell_observed"]
                        )
                    persist()
                    if not accepted:
                        row["termination_reason"] = "execution_rejected"
                        break
                    if score["success"]:
                        row["termination_reason"] = "success"
                        break
                row["status"] = "completed"
                row.setdefault("termination_reason", "command_budget")
            except Exception as error:
                row.update(
                    status="incomplete",
                    error=f"{type(error).__name__}: {error}",
                    unavailable_reason="incomplete_command_prefix",
                )
                raise
            finally:
                persist()
        evaluator.verify_plan_inputs(control, source_root=runtime)
        if identities(args) != expected:
            raise ValueError("identity mismatch after execution")
        report.update(status="completed", integrity_verified_after=True)
    except Exception as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        if robot is not None:
            robot.stop("arrival_feedback_audit_end")
            robot.close()
        persist()
    return 0 if report["status"] == "completed" else 2


def finalize(args, code, timeout):
    path = args.output / "report.json"
    report = json.loads(path.read_text()) if path.exists() else dict(records=planned())
    journal_valid = True
    for row in report["records"]:
        try:
            results, uncertain = recover_trace(args.output / row["kind"] / "trace.jsonl")
        except (ValueError, OSError, KeyError) as error:
            results, uncertain = [], True
            journal_valid = False
            row.update(status="incomplete", journal_error=str(error))
        row["command_slots"] = [
            dict(
                index=i,
                status=("accepted" if results[i]["executed"] else "rejected")
                if i < len(results)
                else ("execution_uncertain" if i == len(results) and uncertain else "not_started"),
            )
            for i in range(16)
        ]
        if row["status"] in ("running", "incomplete"):
            row.update(
                status="incomplete",
                executed_steps=sum(r["executed"] for r in results),
                confirmed_command_results=len(results),
                possible_unobserved_execution=uncertain,
                unavailable_reason="interrupted_prefix",
            )
            if not row.get("advanced_past_goal2"):
                row["advanced_past_goal2"] = None
        elif row["status"] == "not_started":
            row.update(status="not_started_budget" if timeout else "not_started_failure")
    valid = (
        code == 0
        and not timeout
        and report.get("integrity_verified_after") is True
        and journal_valid
    )
    report.update(
        status="completed" if valid else "incomplete",
        returncode=code,
        supervisor_timeout=timeout,
        supervisor_wall_seconds=elapsed(),
    )
    write(path, report)
    return 0 if valid else 2


def supervise_child(child, *, deadline=57, clock=elapsed, sleep=time.sleep, kill=os.killpg):
    """A late observed zero exit is censored too; suspension consumes wall budget."""
    while True:
        timed_out = clock() >= deadline
        returncode = child.poll()
        if timed_out:
            if returncode is None:
                try:
                    kill(child.pid, 9)
                except ProcessLookupError:
                    pass
            return child.wait(), True
        if returncode is not None:
            return returncode, False
        sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wall", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mono", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("repository", "original", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.worker:
        return worker(args)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit")
    args.output.mkdir(parents=True)
    write(
        args.output / "plan.json",
        dict(records=planned(), root_step=ROOT_STEP, max_wall_seconds=60, max_cpu_seconds=40),
    )
    try:
        identity = identities(args)
        load_trace(args.original)
        write(
            args.output / "plan.json",
            dict(
                records=planned(),
                root_step=ROOT_STEP,
                commands_per_sibling=16,
                max_wall_seconds=60,
                max_cpu_seconds=40,
                worker_kill_seconds=57,
                identities=identity,
                environment=dict(
                    python=sys.version,
                    platform=platform.platform(),
                    packages={
                        n: importlib.metadata.version(n) for n in ("torch", "numpy", "mujoco")
                    },
                ),
                no_test_decode=True,
            ),
        )
    except Exception as error:
        write(args.output / "report.json", dict(records=planned(), error=str(error)))
        return finalize(args, 2, False)
    if elapsed() >= 57:
        return finalize(args, 2, True)
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
    with (args.output / "worker.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=args.repository,
            env=os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code, timeout = supervise_child(child)
    return finalize(args, code, timeout)


if __name__ == "__main__":
    raise SystemExit(main())
