"""Bounded, reproducible training on sealed train/validation episode splits only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from embodied_jepa.config import MODELS
from embodied_jepa.contracts import ContractError
from embodied_jepa.data import DatasetStore, concatenate_batches

GIB = 1024**3


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def peak_rss_bytes():
    """Process-lifetime host RSS high-water mark; macOS bytes, Linux KiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return {"revision": revision, "dirty": dirty, "python_source_sha256": digest.hexdigest()}


class BudgetReached(Exception):
    pass


class RunClock:
    """Keep host suspension in budgets without confusing elapsed and CPU time."""

    def __init__(self, *, wall_clock=None, monotonic_clock=None, cpu_clock=None):
        self.wall_clock = wall_clock or time.time
        self.monotonic_clock = monotonic_clock or time.perf_counter
        self.cpu_clock = cpu_clock or time.process_time
        self.wall_start = self.wall_clock()
        self.monotonic_start = self.monotonic_clock()
        self.cpu_start = self.cpu_clock()

    def snapshot(self):
        wall = self.wall_clock() - self.wall_start
        monotonic = self.monotonic_clock() - self.monotonic_start
        return {
            "elapsed_seconds": max(0.0, wall, monotonic),
            "wall_clock_elapsed_seconds": wall,
            "monotonic_elapsed_seconds": monotonic,
            "process_cpu_seconds": self.cpu_clock() - self.cpu_start,
        }

    def elapsed(self):
        return self.snapshot()["elapsed_seconds"]

    def check(self, max_seconds):
        if self.elapsed() >= max_seconds:
            raise BudgetReached("time_budget")


class EpisodeCache:
    """Decode each selected episode once; compact indices sample windows uniformly."""

    def __init__(self, store, *, memory_limit_bytes, check_budget=lambda: None):
        splits = store.manifest.get("splits")
        if not splits or not splits.get("train") or not splits.get("val"):
            raise ContractError("training requires sealed nonempty train and val splits")
        self.split_ids = {name: tuple(splits[name]) for name in ("train", "val")}
        if set(self.split_ids["train"]) & set(self.split_ids["val"]):
            raise ContractError("training and validation episodes overlap")
        excluded = set(splits.get("test", [])) | set(splits.get("holdout", []))
        if excluded & (set(self.split_ids["train"]) | set(self.split_ids["val"])):
            raise ContractError("training/validation overlaps a test or holdout split")
        rows = {row["episode_id"]: row for row in store.manifest["episodes"]}
        groups = [
            {rows[name]["session_id"] for name in self.split_ids[split]}
            for split in ("train", "val")
        ]
        if groups[0] & groups[1]:
            raise ContractError("train and validation sessions overlap")
        if (groups[0] | groups[1]) & {rows[name]["session_id"] for name in excluded}:
            raise ContractError("train/validation sessions overlap test/holdout")
        info = json.loads((store.root / "meta/info.json").read_text())
        pixel_bytes = sum(
            int(np.prod(feature["shape"]))
            for feature in info["features"].values()
            if feature["dtype"] == "image"
        )
        bytes_per_frame = pixel_bytes + store.state_schema.dimension * 5 + 14 * 8 + 8
        self.bytes_per_frame = bytes_per_frame
        count = sum(rows[name]["length"] for ids in self.split_ids.values() for name in ids)
        self.estimated_bytes = count * bytes_per_frame
        # Reserve room for decoder copies, canonical batches and model state.
        if peak_rss_bytes() + 3 * self.estimated_bytes > memory_limit_bytes:
            raise ContractError(
                "decoded corpus estimate exceeds memory budget; use a smaller corpus"
            )
        self.episodes = {}
        self.decoded_bytes = 0
        for ids in self.split_ids.values():
            for name in ids:
                check_budget()
                episode = store.read_episode(name)
                self.episodes[name] = episode
                self.decoded_bytes += sum(frames.nbytes for frames in episode.observations.values())
                self.decoded_bytes += sum(
                    getattr(episode, field).nbytes
                    for field in ("robot_states", "state_mask", "actions", "timestamps")
                )
                if episode.raw_actions is not None:
                    self.decoded_bytes += episode.raw_actions.nbytes
                if peak_rss_bytes() > memory_limit_bytes:
                    raise BudgetReached("memory_budget")
        self._indices = {}

    def index(self, split, horizon):
        key = split, horizon
        if key not in self._indices:
            names = [
                name for name in self.split_ids[split] if self.episodes[name].transitions >= horizon
            ]
            counts = [self.episodes[name].transitions - horizon + 1 for name in names]
            self._indices[key] = names, np.cumsum(counts, dtype=np.int64)
        return self._indices[key]

    def count(self, split, horizon):
        _, cumulative = self.index(split, horizon)
        return int(cumulative[-1]) if len(cumulative) else 0

    def sample(self, split, horizon, batch_size, rng):
        names, cumulative = self.index(split, horizon)
        if not len(cumulative):
            raise ContractError(f"{split} has no complete horizon-{horizon} windows")
        selected = rng.integers(int(cumulative[-1]), size=batch_size)
        batches = []
        for offset in selected:
            index = int(np.searchsorted(cumulative, offset, side="right"))
            before = int(cumulative[index - 1]) if index else 0
            batches.append(self.episodes[names[index]].sequence(int(offset) - before, horizon))
        return concatenate_batches(batches)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def selection_decision(diagnostics, selection, training_horizon):
    """Apply a declared validation-only selector; never silently substitute a state."""
    horizon = 4 if selection == "noncollapsed_relative" else training_horizon
    result = {
        "method": selection,
        "metric_definition_version": 2,
        "horizon": horizon,
        "eligible": False,
        "score": None,
        "rejection_reasons": [],
    }
    measured = diagnostics.get(str(horizon), {})
    if measured.get("status") != "measured":
        result["rejection_reasons"].append("required_validation_horizon_unavailable")
        return result
    metrics = measured["metrics"]
    result["metric_definition_version"] = metrics.get("metric_definition_version", 2)
    if selection == "noncollapsed_relative":
        if metrics.get("metric_definition_version") != 2:
            result["rejection_reasons"].append("unsupported_metric_definition_version")
            return result
        for space in ("online", "target"):
            if metrics[f"{space}_collapsed_fraction"] > 0.05:
                result["rejection_reasons"].append(f"{space}_collapsed_fraction_above_0.05")
            if metrics[f"{space}_latent_std_mean"] < 0.1:
                result["rejection_reasons"].append(f"{space}_latent_std_mean_below_0.1")
        score = metrics["prediction_mse"] / max(metrics["persistence_mse"], 1e-12)
    else:
        score = metrics["prediction_mse"]
    if not np.isfinite(score):
        raise ContractError("non-finite checkpoint selection score")
    result["score"] = score
    result["eligible"] = not result["rejection_reasons"]
    return result


def train(
    dataset,
    backend,
    output,
    *,
    steps=500,
    batch_size=16,
    horizon=4,
    seed=0,
    device="cpu",
    max_seconds=600.0,
    validation_every=50,
    validation_batches=4,
    model_config=None,
    memory_limit_gib=16.0,
    selection="raw_mse",
):
    """Train a fixed budget; `output` is best-by-validation, `.latest.pt` is last state."""
    import torch

    for name, value in dict(
        steps=steps,
        batch_size=batch_size,
        horizon=horizon,
        validation_every=validation_every,
        validation_batches=validation_batches,
    ).items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    if not np.isfinite(memory_limit_gib) or not 0 < memory_limit_gib <= 16:
        raise ValueError("memory_limit_gib must lie in (0,16]")
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS unavailable")
    if selection not in ("raw_mse", "noncollapsed_relative"):
        raise ValueError("selection must be raw_mse or noncollapsed_relative")
    MODELS.require(backend)
    output = Path(output)
    paths = {
        "best": output,
        "latest": output.with_name(output.stem + ".latest.pt"),
        "report": output.with_name(output.stem + ".run.json"),
        "curves": output.with_name(output.stem + ".metrics.jsonl"),
    }
    if len(set(paths.values())) != len(paths) or any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite existing training artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = source_identity()
    clock = RunClock()
    memory_limit = int(memory_limit_gib * GIB)
    sampler = np.random.default_rng(seed)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(min(original_threads, 4))
    model, cache, best_step, best_error, completed = None, None, None, None, 0
    best_score = None
    report = {
        "format_version": 1,
        "status": "running",
        "backend": backend,
        "dataset": str(Path(dataset).resolve()),
        "device": device,
        "seed": seed,
        "selection": {
            "method": selection,
            "metric_definition_version": 2,
            "horizon": 4 if selection == "noncollapsed_relative" else horizon,
            "eligibility": {
                "spaces": ["online", "target"],
                "maximum_collapsed_fraction": 0.05,
                "minimum_latent_std_mean": 0.1,
            }
            if selection == "noncollapsed_relative"
            else None,
        },
        "budget": {
            "steps": steps,
            "batch_size": batch_size,
            "horizon": horizon,
            "max_seconds": max_seconds,
            "memory_limit_bytes": memory_limit,
            "validation_every": validation_every,
            "validation_batches": validation_batches,
        },
        "source": source,
        "model_config_overrides": model_config or {},
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "validation": [],
        "test_samples_loaded": False,
        "timing_definitions": {
            "elapsed_seconds": "max(wall-clock delta, monotonic delta, 0); budget clock",
            "wall_clock_elapsed_seconds": "time.time delta; includes host suspension/clock changes",
            "monotonic_elapsed_seconds": "perf_counter delta; may exclude host suspension",
            "process_cpu_seconds": "process_time delta; aggregate process CPU, not elapsed time",
            "update_seconds": "synchronized perf_counter delta around a model update",
        },
    }
    for package in ("transformers", "einops", "pyarrow", "pandas", "pillow"):
        try:
            report["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    _write_json(paths["report"], report)

    def synchronize():
        if device == "mps":
            torch.mps.synchronize()

    def check_budget():
        clock.check(max_seconds)
        if peak_rss_bytes() > memory_limit:
            raise BudgetReached("memory_budget")

    def record(item):
        with paths["curves"].open("a") as stream:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")

    def checkpoint(path):
        model.metadata["runner_state"] = {
            "step": completed,
            "sampler_rng": sampler.bit_generator.state,
            "sampling": "uniform_train_windows_with_replacement",
            "validation_cohort_seed": seed + 10001,
            "selection": report["selection"],
        }
        model.save(path)

    def validate():
        nonlocal best_step, best_error, best_score
        diagnostics = {}
        for length in dict.fromkeys((horizon, 1, 4, 8)):
            if length > model.capabilities.max_horizon or not cache.count("val", length):
                diagnostics[str(length)] = {"status": "unavailable", "reason": "no valid windows"}
                continue
            rng = np.random.default_rng(seed + 10001 + length)
            metrics = []
            cohort = hashlib.sha256()
            for _ in range(validation_batches):
                check_budget()
                batch = cache.sample("val", length, batch_size, rng)
                cohort.update(json.dumps(batch.episode_ids).encode())
                cohort.update(batch.timestamps.tobytes())
                synchronize()
                measured = model.diagnostics(batch)
                synchronize()
                if not all(np.isfinite(value) for value in measured.values()):
                    raise ContractError("non-finite validation metrics")
                metrics.append(measured)
            diagnostics[str(length)] = {
                "status": "measured",
                "windows": batch_size * validation_batches,
                "cohort_sha256": cohort.hexdigest(),
                "metrics": {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
            }
        error = diagnostics[str(horizon)]["metrics"]["prediction_mse"]
        decision = selection_decision(diagnostics, selection, horizon)
        report["selection"]["metric_definition_version"] = decision["metric_definition_version"]
        improved = decision["eligible"] and (best_score is None or decision["score"] < best_score)
        decision["selected"] = improved
        decision["selection_reason"] = (
            "ineligible"
            if not decision["eligible"]
            else "new_best"
            if improved
            else "no_improvement"
        )
        event = {
            "kind": "validation",
            "metric_definition_version": decision["metric_definition_version"],
            "step": completed,
            "elapsed_seconds": clock.elapsed(),
            "selection_horizon": decision["horizon"],
            "selection": decision,
            "prediction_mse": error,
            "horizons": diagnostics,
        }
        report["validation"].append(event)
        record(event)
        if improved:
            best_score, best_step = decision["score"], completed
            best_error = diagnostics[str(decision["horizon"])]["metrics"]["prediction_mse"]
            checkpoint(paths["best"])
        _write_json(paths["report"], report)

    failed = None
    try:
        check_budget()
        store = DatasetStore(dataset)
        metadata = {
            "dataset_hash": store.manifest_hash,
            "split_hash": json_hash(
                {
                    "splits": store.manifest.get("splits"),
                    "policy": store.manifest.get("split_policy"),
                }
            ),
            "action_hash": json_hash(store.manifest["action_manifest"]),
            "source_revision": source["revision"],
            "source_tree_hash": source["python_source_sha256"],
        }
        report["provenance"] = metadata.copy()
        report["dataset_provenance"] = store.manifest["provenance"]
        cache = EpisodeCache(store, memory_limit_bytes=memory_limit, check_budget=check_budget)
        for split in ("train", "val"):
            if not cache.count(split, horizon):
                raise ContractError(f"{split} has no windows at configured horizon {horizon}")
        report["cache"] = {
            "decoded_bytes": cache.decoded_bytes,
            "episode_ids": cache.split_ids,
            "train_windows": cache.count("train", horizon),
            "val_windows": cache.count("val", horizon),
        }
        # Bound canonical copies and float image conversion before allocating a batch.
        max_window = max(length for length in (horizon, 1, 4, 8) if cache.count("val", length))
        batch_reserve = 16 * batch_size * (max_window + 1) * cache.bytes_per_frame
        if peak_rss_bytes() + batch_reserve > memory_limit:
            raise ContractError("batch estimate exceeds memory budget; reduce batch size")
        check_budget()
        model = MODELS.create(
            backend,
            state_schema=store.state_schema,
            device=device,
            seed=seed,
            config=model_config,
            metadata=metadata,
        )
        model.capabilities.require(
            action_schema=model.capabilities.action_schema,
            state_schema=store.state_schema,
            horizon=horizon,
            device=device,
        )
        report["model_config"] = model.config
        report["model_implementation_sha256"] = model.implementation_sha256
        report["model_source_revision"] = model.source_revision
        fit_normalization = getattr(model, "fit_normalization", None)
        if callable(fit_normalization):
            # Optional model-owned preprocessing sees only sealed training episodes.
            # Stream bounded chunks, including every transition exactly once.
            def normalization_batches():
                for name in cache.split_ids["train"]:
                    episode = cache.episodes[name]
                    chunk = min(64, model.capabilities.max_horizon)
                    for start in range(0, episode.transitions, chunk):
                        check_budget()
                        yield episode.sequence(start, min(chunk, episode.transitions - start))

            fit_normalization(
                normalization_batches(), training_episode_ids=cache.split_ids["train"]
            )
            report["normalization"] = {
                "episode_ids": list(cache.split_ids["train"]),
                "source": "sealed training split only",
                "transitions": sum(
                    cache.episodes[name].transitions for name in cache.split_ids["train"]
                ),
            }
            check_budget()
        validate()  # Untrained baseline can legitimately remain best after a negative run.
        for step in range(1, steps + 1):
            check_budget()
            batch = cache.sample("train", horizon, batch_size, sampler)
            synchronize()
            update_start = time.perf_counter()
            metrics = model.train_step(batch)
            synchronize()
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ContractError("non-finite training metrics")
            completed = step
            record(
                {
                    "kind": "train",
                    "step": step,
                    "metrics": dict(metrics),
                    "update_seconds": time.perf_counter() - update_start,
                    "elapsed_seconds": clock.elapsed(),
                }
            )
            if step % validation_every == 0 or step == steps:
                validate()
        report["status"] = "completed" if best_step is not None else "selection_failed"
        if best_step is None:
            report["error"] = "No validation checkpoint met the declared eligibility criteria"
    except BudgetReached as error:
        report["status"] = str(error)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        failed = error
    finally:
        try:
            if model is not None:
                try:
                    checkpoint(paths["latest"])
                except Exception as error:
                    # A model can fail before it has a valid fitted preprocessing
                    # state. Preserve that failure and the report even if no
                    # loadable latest checkpoint can be written.
                    report["latest_checkpoint_error"] = f"{type(error).__name__}: {error}"
                    if failed is None and report["status"] == "completed":
                        failed = error
                        report["status"] = "failed"
            synchronize()
            report.update(
                completed_steps=completed,
                best_step=best_step,
                best_validation_mse=best_error,
                best_selection_score=best_score,
                **clock.snapshot(),
                peak_host_rss_bytes=peak_rss_bytes(),
                rss_scope="process-lifetime host RSS high-water mark",
                sampler_rng=sampler.bit_generator.state,
            )
            if device == "mps":
                report["final_mps_allocated_bytes"] = torch.mps.current_allocated_memory()
                report["final_mps_driver_bytes"] = torch.mps.driver_allocated_memory()
            report["best_checkpoint_exists"] = paths["best"].exists()
            report["latest_checkpoint_exists"] = paths["latest"].exists()
            _write_json(paths["report"], report)
        finally:
            torch.set_num_threads(original_threads)
    if failed is not None:
        raise failed
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("native_jepa", "leworldmodel", "sensor_wm"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument(
        "--model-config", type=Path, help="JSON mapping of backend config overrides"
    )
    parser.add_argument("--memory-limit-gib", type=float, default=16)
    parser.add_argument(
        "--selection", choices=("raw_mse", "noncollapsed_relative"), default="raw_mse"
    )
    args = vars(parser.parse_args())
    if args["model_config"] is not None:
        args["model_config"] = json.loads(args["model_config"].read_text())
    report = train(**args)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "completed_steps",
                    "best_step",
                    "best_validation_mse",
                    "artifacts",
                )
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
