"""Strict validation for newly produced common-mode benchmark records.

Historical pilot records retain their original contents; records missing later
required outcome fields are historical evidence, not silently upgraded records.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime

from embodied_jepa.planning import CEMConfig

STAGES = ("reach", "grasp", "transport", "place", "release")
BACKENDS = {"native_jepa", "leworldmodel", "control:hold", "control:random", "control:oracle"}
TERMINATIONS = {
    "success",
    "step_limit",
    "stopped",
    "rejected",
    "deadline_miss",
    "stale_observation",
    "runtime_error",
    "unreachable_goal",
}


def _require(condition, path, message):
    if not condition:
        raise ValueError(f"{path}: {message}")


def _mapping(value, path):
    _require(isinstance(value, dict), path, "must be an object")
    return value


def _text(value, path):
    _require(isinstance(value, str) and bool(value.strip()), path, "must be a nonempty string")


def _integer(value, path, minimum=0):
    _require(type(value) is int and value >= minimum, path, f"must be an integer >= {minimum}")


def _number(value, path, *, nonnegative=True):
    _require(
        type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0),
        path,
        "must be a finite number" + (" >= 0" if nonnegative else ""),
    )


def _hash(value, path):
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        path,
        "must be a lowercase SHA-256 digest",
    )


def _action(value, path):
    _require(isinstance(value, (list, tuple)) and len(value) == 14, path, "must be a 14D action")
    for index, entry in enumerate(value):
        _number(entry, f"{path}[{index}]", nonnegative=False)
        _require(-1.000001 <= entry <= 1.000001, path, "action is outside normalized bounds")


def _score(score, path):
    _mapping(score, path)
    _require(type(score.get("success")) is bool, path, "requires an explicit success outcome")
    for stage in STAGES:
        _require(stage in score, path, f"missing stage {stage}")
        value = score[stage]
        _require(value is None or type(value) is bool, f"{path}.{stage}", "must be bool or null")
        if value is None:
            _text(score.get("missing_stage_reason"), f"{path}.missing_stage_reason")
    for name, value in score.items():
        if name.endswith(("_m", "_s")) and value is not None:
            # Heights can legitimately be negative after a dropped object.
            _number(value, f"{path}.{name}", nonnegative=not name.endswith("height_m"))


def validate_results(run_manifest, episodes):
    """Reject malformed metadata, omitted failures, or inconsistent command traces."""
    run = _mapping(run_manifest, "run")
    required = {
        "schema_version",
        "run_id",
        "timestamp",
        "source_revision",
        "backend",
        "checkpoint_hash",
        "mode",
        "dataset_hash",
        "split_hash",
        "action_hash",
        "environment",
        "planner",
        "train_seed",
        "evaluation_seeds",
        "task_version",
        "timeout_seconds",
        "model_diagnostics",
    }
    _require(
        not (required - run.keys()), "run", f"missing run fields: {sorted(required - run.keys())}"
    )
    _require(
        type(run["schema_version"]) is int and run["schema_version"] == 1,
        "run.schema_version",
        "unsupported schema",
    )
    _require(run["mode"] == "common", "run.mode", "unsupported mode")
    for name in ("run_id", "task_version"):
        _text(run[name], f"run.{name}")
    _text(run["timestamp"], "run.timestamp")
    try:
        stamp = datetime.fromisoformat(run["timestamp"])
    except ValueError as error:
        raise ValueError("run.timestamp: invalid ISO timestamp") from error
    _require(stamp.utcoffset() is not None, "run.timestamp", "timezone is required")
    _require(
        isinstance(run["source_revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", run["source_revision"]) is not None,
        "run.source_revision",
        "must identify a full Git revision",
    )
    _require(
        isinstance(run["backend"], str) and run["backend"] in BACKENDS,
        "run.backend",
        "unknown backend",
    )
    control = run["backend"].startswith("control:")
    for name in ("dataset_hash", "split_hash", "action_hash"):
        _hash(run[name], f"run.{name}")
    if control:
        _require(
            run["checkpoint_hash"] is None and run["train_seed"] is None,
            "run",
            "control policy must not claim a model checkpoint/train seed",
        )
        _text(run.get("checkpoint_missing_reason"), "run.checkpoint_missing_reason")
        _text(run.get("train_seed_missing_reason"), "run.train_seed_missing_reason")
    else:
        _hash(run["checkpoint_hash"], "run.checkpoint_hash")
        _integer(run["train_seed"], "run.train_seed")
    for name, value in run.items():
        if name.endswith(("_hash", "_sha256")) and value is not None:
            _hash(value, f"run.{name}")
    if "implementation_hashes" in run:
        hashes = _mapping(run["implementation_hashes"], "run.implementation_hashes")
        _require(bool(hashes), "run.implementation_hashes", "must not be empty")
        for name, value in hashes.items():
            _text(name, "run.implementation_hashes key")
            _hash(value, f"run.implementation_hashes.{name}")
    if "source_dirty" in run:
        _require(type(run["source_dirty"]) is bool, "run.source_dirty", "must be boolean")
    if "training_report" in run:
        if run["training_report"] is None:
            _text(run.get("training_report_missing_reason"), "run.training_report_missing_reason")
        else:
            report = _mapping(run["training_report"], "run.training_report")
            _text(report.get("path"), "run.training_report.path")
            _hash(report.get("sha256"), "run.training_report.sha256")
    env = _mapping(run["environment"], "run.environment")
    for name in ("platform", "python", "engine", "torch", "numpy"):
        _text(env.get(name), f"run.environment.{name}")
    _require(env.get("device") in ("cpu", "mps"), "run.environment.device", "unknown device")
    _require(env["engine"] == "mujoco", "run.environment.engine", "unsupported engine")
    _text(env.get("mujoco"), "run.environment.mujoco")
    planner = _mapping(run["planner"], "run.planner")
    names = set(CEMConfig.__dataclass_fields__)
    # Schema-v1 archives predate this opt-in flag; omission means legacy false.
    _require(
        set(planner) in (names, names - {"project_candidates"}),
        "run.planner",
        "requires the complete resolved CEM budget",
    )
    for name in ("action_penalty", "minimum_std"):
        _number(planner[name], f"run.planner.{name}")
    for name in ("lower_bounds", "upper_bounds"):
        _action(planner[name], f"run.planner.{name}")
    try:
        budget = CEMConfig(**planner)
    except (TypeError, ValueError) as error:
        raise ValueError(f"run.planner: {error}") from error
    _number(run["timeout_seconds"], "run.timeout_seconds")
    _require(run["timeout_seconds"] > 0, "run.timeout_seconds", "must be positive")
    if "max_steps" in run:
        _integer(run["max_steps"], "run.max_steps", 1)
    seeds = run["evaluation_seeds"]
    _require(isinstance(seeds, list) and bool(seeds), "run.evaluation_seeds", "requires seeds")
    for seed in seeds:
        _integer(seed, "run.evaluation_seeds")
    _require(len(set(seeds)) == len(seeds), "run.evaluation_seeds", "duplicate seeds")
    if run["model_diagnostics"] is None:
        _text(run.get("model_diagnostics_reason"), "run.model_diagnostics_reason")
    else:
        _mapping(run["model_diagnostics"], "run.model_diagnostics")
    _require(isinstance(episodes, list), "episodes", "must be a list")
    for episode in episodes:
        _mapping(episode, "episode")
        _integer(episode.get("seed"), "episode.seed")
    _require(
        [e["seed"] for e in episodes] == seeds,
        "episodes",
        "all frozen evaluation seeds must be recorded exactly once in order",
    )
    for episode in episodes:
        path = f"episode[{episode['seed']}]"
        _score(episode.get("score"), f"{path}.score")
        reason = episode.get("termination_reason")
        _require(isinstance(reason, str) and reason in TERMINATIONS, path, "unknown termination")
        _require(
            reason != "success" or episode["score"]["success"],
            path,
            "success termination conflicts with scoring evidence",
        )
        for name in ("executed_steps", "replans"):
            _integer(episode.get(name), f"{path}.{name}")
        _number(episode.get("wall_seconds"), f"{path}.wall_seconds")
        _require("collision_violations" in episode, path, "missing collision measurement")
        if episode["collision_violations"] is None:
            _text(episode.get("collision_missing_reason"), f"{path}.collision_missing_reason")
        else:
            _integer(episode["collision_violations"], f"{path}.collision_violations")
        if "limit_rejections" in episode:
            _integer(episode["limit_rejections"], f"{path}.limit_rejections")
        for name in ("goal_image_sha256", "initial_image_sha256"):
            if name in episode:
                _hash(episode[name], f"{path}.{name}")
        traces = episode.get("trace")
        _require(isinstance(traces, list), f"{path}.trace", "must be a list")
        previous_step = -1
        for index, trace in enumerate(traces):
            tp = f"{path}.trace[{index}]"
            _mapping(trace, tp)
            _integer(trace.get("step"), f"{tp}.step")
            _require(
                previous_step <= trace["step"] <= previous_step + 1,
                tp,
                "trace steps must be ordered without gaps",
            )
            previous_step = trace["step"]
            if "max_steps" in run:
                _require(trace["step"] < run["max_steps"], tp, "step exceeds run budget")
            executed = trace.get("executed")
            _require(
                "executed" in trace and (executed is None or type(executed) is bool),
                tp,
                "executed must be bool or explicitly uncertain null",
            )
            if executed is None:
                _require(
                    trace.get("execution_uncertain") is True and bool(trace.get("error")),
                    tp,
                    "null execution requires an uncertainty/error record",
                )
            if "status" in trace:
                _require(
                    trace["status"] in ("applied", "clipped", "rejected", "stopped"),
                    tp,
                    "unknown execution status",
                )
                _require(
                    (trace["status"] in ("applied", "clipped")) == (executed is True),
                    tp,
                    "execution status conflicts with executed flag",
                )
            if executed is True:
                _action(trace.get("action"), f"{tp}.action")
            elif trace.get("action") is not None:
                raise ValueError(f"{tp}: unapplied command cannot have an applied action")
            if "requested_action" in trace:
                _action(trace["requested_action"], f"{tp}.requested_action")
            for name in ("sampled_action", "projected_action"):
                if name in trace:
                    _action(trace[name], f"{tp}.{name}")
            if budget.project_candidates and "requested_action" in trace:
                _action(trace.get("sampled_action"), f"{tp}.sampled_action")
                _action(trace.get("projected_action"), f"{tp}.projected_action")
                _require(
                    trace["projected_action"] == trace["requested_action"],
                    tp,
                    "execution request must equal model-scored projected action",
                )
            if budget.project_candidates and "candidate_evaluations" in trace:
                _number(trace.get("projection_seconds"), f"{tp}.projection_seconds")
                _integer(trace.get("feasible_candidates"), f"{tp}.feasible_candidates", 1)
                _require(
                    trace["feasible_candidates"] <= trace["candidate_evaluations"],
                    tp,
                    "feasible count exceeds evaluated candidates",
                )
            for name in (
                "planning_seconds",
                "control_seconds",
                "observation_timestamp",
                "execution_timestamp",
            ):
                if name in trace:
                    _number(trace[name], f"{tp}.{name}")
            if "planned_cost" in trace:
                _number(trace["planned_cost"], f"{tp}.planned_cost", nonnegative=False)
            if "candidate_evaluations" in trace:
                _integer(trace["candidate_evaluations"], f"{tp}.candidate_evaluations", 1)
                _require(
                    not control
                    and trace["candidate_evaluations"] == budget.samples * budget.iterations,
                    tp,
                    "candidate count conflicts with resolved planner budget",
                )
                _number(trace.get("planning_seconds"), f"{tp}.planning_seconds")
            if executed is True and not control:
                _require(
                    "candidate_evaluations" in trace,
                    tp,
                    "model execution requires planning evidence",
                )
                _number(trace.get("control_seconds"), f"{tp}.control_seconds")
                _require(
                    trace["control_seconds"] <= run["timeout_seconds"],
                    tp,
                    "executed command exceeds control deadline",
                )
            if control:
                _require(
                    "planning_seconds" not in trace, tp, "control policy must not claim planning"
                )
        _require(
            episode["executed_steps"] == sum(t["executed"] is True for t in traces),
            path,
            "executed step count conflicts with trace",
        )
        _require(
            episode["replans"] == sum("candidate_evaluations" in t for t in traces),
            path,
            "replan count conflicts with trace",
        )
    try:
        json.dumps({"run": run, "episodes": episodes}, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"results must contain finite JSON values: {error}") from error
