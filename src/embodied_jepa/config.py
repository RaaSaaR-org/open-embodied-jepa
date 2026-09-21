"""Strict, inherited experiment configuration with lazy component selection."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from embodied_jepa.planning import CEMConfig
from embodied_jepa.registry import EMBODIMENTS, MODELS, PLANNERS, TASKS

MODELS.register("native_jepa", "embodied_jepa.models:NativeJEPA")
MODELS.register("leworldmodel", "embodied_jepa.models:LeWM")
MODELS.register("sensor_wm", "embodied_jepa.models.sensor:SensorWorldModel")
MODELS.register(
    "aligned_sensor_wm_v1", "embodied_jepa.models.aligned_sensor:AlignedSensorWorldModel"
)
MODELS.register("jepa_wms", "embodied_jepa.models:JEPAWMs")
EMBODIMENTS.register("unitree_g1_dex3", "embodied_jepa.embodiment:G1Embodiment")
PLANNERS.register("cem", "embodied_jepa.planning:CEMPlanner")
TASKS.register("reach", "embodied_jepa.task:ReachTask")
TASKS.register("apple_to_plate", "embodied_jepa.task:AppleToPlateTask")


def _mapping(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if set(value) - set(allowed):
        raise ValueError(f"unknown {label} keys: {sorted(set(value) - set(allowed))}")
    return value


def _merge(base, child):
    result = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError("configuration inheritance cycle")
    seen.add(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    # Resolve each declared path at its defining file before inheritance merges it.
    for section, key in (
        ("dataset", "root"),
        ("embodiment", "action_manifest"),
        ("evaluation", "output_root"),
        ("task", "goal_manifest"),
    ):
        group = config.get(section)
        if isinstance(group, dict) and isinstance(group.get(key), str) and group[key].strip():
            group[key] = str((path.parent / group[key]).resolve())
    model = config.get("world_model")
    if isinstance(model, dict) and isinstance(model.get("checkpoints"), dict):
        model["checkpoints"] = {
            key: str((path.parent / value).resolve())
            if isinstance(value, str) and value.strip()
            else value
            for key, value in model["checkpoints"].items()
        }
    parent = config.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError("extends must name a configuration file")
        config = _merge(_read(path.parent / parent, seen), config)
    return config


def _path(value, base: Path, label: str, *, must_exist=True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a resolved nonempty path")
    path = (base / value).resolve()
    if must_exist and not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if (
        must_exist
        and label in ("checkpoint", "action manifest", "goal manifest")
        and not path.is_file()
    ):
        raise ValueError(f"{label} must be a file")
    if must_exist and label == "dataset root" and not path.is_dir():
        raise ValueError("dataset root must be a directory")
    return path


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str
    device: str
    seed: int
    dataset_root: Path
    action_manifest: Path
    checkpoint: Path
    model_settings: dict[str, Any]
    planner: CEMConfig
    timeout_seconds: float
    task: str
    max_steps: int
    evaluation_seeds: tuple[int, ...]
    output_root: Path
    resolved: dict[str, Any]
    goal_manifest: Path | None = None

    @classmethod
    def load(cls, path: str | Path, *, require_checkpoint=True):
        path = Path(path).resolve()
        raw = _read(path, set())
        _mapping(
            raw,
            (
                "schema_version",
                "mode",
                "seed",
                "runtime",
                "world_model",
                "embodiment",
                "dataset",
                "planner",
                "task",
                "evaluation",
            ),
            "experiment",
        )
        if raw.get("schema_version") != 0 or type(raw.get("schema_version")) is not int:
            raise ValueError("schema_version must be 0")
        if raw.get("mode", "common") != "common":
            raise ValueError("only common mode is implemented")
        seed = raw.get("seed", 0)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        runtime = _mapping(raw.get("runtime", {}), ("device",), "runtime")
        device = runtime.get("device", "cpu")
        if device not in ("cpu", "mps", "auto"):
            raise ValueError("local device must be cpu, mps or auto")
        if device in ("auto", "mps"):
            import torch

            available = torch.backends.mps.is_available()
            if device == "mps" and not available:
                raise ValueError("requested MPS is unavailable")
            device = "mps" if available else "cpu"
        model = _mapping(
            raw.get("world_model", {}),
            ("backend", "checkpoints", "settings", "history"),
            "world_model",
        )
        backend = model.get("backend")
        MODELS.require(backend)
        if type(model.get("history", 1)) is not int or model.get("history", 1) != 1:
            raise ValueError("current observation interface requires history=1")
        checkpoints = _mapping(
            model.get("checkpoints", {}),
            ("native_jepa", "leworldmodel", "jepa_wms", "sensor_wm", "aligned_sensor_wm_v1"),
            "checkpoints",
        )
        checkpoint = _path(
            checkpoints.get(backend), path.parent, "checkpoint", must_exist=require_checkpoint
        )
        settings = _mapping(
            model.get("settings", {}),
            ("native_jepa", "leworldmodel", "jepa_wms", "sensor_wm", "aligned_sensor_wm_v1"),
            "model settings",
        )
        selected_settings = settings.get(backend, {})
        if not isinstance(selected_settings, dict):
            raise ValueError("backend settings must be a mapping")
        embodiment = _mapping(
            raw.get("embodiment", {}),
            ("backend", "transport", "action_manifest", "action_schema"),
            "embodiment",
        )
        EMBODIMENTS.require(embodiment.get("backend"))
        if embodiment.get("transport") != "mujoco":
            raise ValueError("only MuJoCo execution is enabled")
        if embodiment.get("action_schema", "ee_delta_grasp_v0") != "ee_delta_grasp_v0":
            raise ValueError("unsupported action schema")
        action_manifest = _path(embodiment.get("action_manifest"), path.parent, "action manifest")
        manifest = json.loads(action_manifest.read_text())
        if not isinstance(manifest, dict) or manifest.get("action_schema") != "ee_delta_grasp_v0":
            raise ValueError("action manifest has incompatible schema")
        for field in (
            "translation_per_step_m",
            "rotation_per_step_rad",
            "control_dt_s",
            "joint_speed_limit_rad_s",
        ):
            value = manifest.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"unresolved or invalid action manifest {field}")
        dataset = _mapping(raw.get("dataset", {}), ("root",), "dataset")
        dataset_root = _path(dataset.get("root"), path.parent, "dataset root")
        planner = dict(
            _mapping(
                raw.get("planner", {}),
                (
                    "backend",
                    "project_candidates",
                    "horizon",
                    "samples",
                    "iterations",
                    "elites",
                    "action_penalty",
                    "minimum_std",
                    "lower_bounds",
                    "upper_bounds",
                    "timeout_seconds",
                ),
                "planner",
            )
        )
        PLANNERS.require(planner.pop("backend", "cem"))
        timeout = planner.pop("timeout_seconds", 5.0)

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (float, int))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        budget = CEMConfig(**planner, seed=seed)
        task = _mapping(raw.get("task", {}), ("name", "max_steps", "goal_manifest"), "task")
        TASKS.require(task.get("name"))
        goal_manifest = (
            _path(task["goal_manifest"], path.parent, "goal manifest")
            if "goal_manifest" in task
            else None
        )
        steps = task.get("max_steps", 50)
        if type(steps) is not int or steps < 1:
            raise ValueError("max_steps must be positive")
        evaluation = _mapping(raw.get("evaluation", {}), ("seeds", "output_root"), "evaluation")
        seeds = evaluation.get("seeds", [0, 1, 2, 3, 4])
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("evaluation seeds must be distinct nonnegative integers")
        output = _path(
            evaluation.get("output_root", "../outputs"),
            path.parent,
            "output root",
            must_exist=False,
        )
        planner_config = asdict(budget)
        planner_config.pop("seed")  # The experiment seed is the sole CEM seed source.
        resolved = {
            "schema_version": 0,
            "mode": "common",
            "seed": seed,
            "runtime": {"device": device},
            "world_model": {
                "backend": backend,
                "checkpoints": checkpoints,
                "settings": settings,
                "history": 1,
            },
            "embodiment": {
                "backend": embodiment["backend"],
                "transport": "mujoco",
                "action_schema": "ee_delta_grasp_v0",
                "action_manifest": str(action_manifest),
            },
            "dataset": {"root": str(dataset_root)},
            "planner": {"backend": "cem", **planner_config, "timeout_seconds": timeout},
            "task": {
                "name": task["name"],
                "max_steps": steps,
                **({"goal_manifest": str(goal_manifest)} if goal_manifest is not None else {}),
            },
            "evaluation": {"seeds": seeds, "output_root": str(output)},
        }
        return cls(
            backend,
            device,
            seed,
            dataset_root,
            action_manifest,
            checkpoint,
            selected_settings,
            budget,
            float(timeout),
            task["name"],
            steps,
            tuple(seeds),
            output,
            resolved,
            goal_manifest=goal_manifest,
        )

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.resolved, indent=2, sort_keys=True) + "\n")
