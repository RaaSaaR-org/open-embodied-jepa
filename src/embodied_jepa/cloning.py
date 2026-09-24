"""Behaviour cloning on the wide Apple corpus, over a frozen encoder (TASK-056).

Implements stage 2 of `docs/experiments/apple_policy_v1.md`: build the BC cohort under the
protocol's exclusion rules, precompute frozen image features once, and train the arms.

**The target is ``collector__base_action``**, the *unperturbed* scripted command recorded at
every visited state -- not the executed action. 150 of the corpus's 200 roots were driven with
injected OU noise, so the pairing is (state visited under noise, label the clean expert would
have commanded): noise-injected behaviour cloning, which is the standard remedy for BC's
compounding error and which this corpus happens to provide because it was collected for
world-model reasons.

``collector__*`` is a **privileged training label** (``training_labels``): the collector read
simulator truth at reset. It is a target and nothing else -- never a model input, a controller
input, or a scoring quantity -- and loading it requires the same explicit acknowledgement the
readout targets already require.

Three exclusions, from §2.2 of the protocol. Two of them would silently poison the run:

* **aim-offset roots** -- every fifth root's script aims 1.5-3.0 cm off the apple *by design*,
  so its base action is self-consistent with a wrong target. As BC labels that is 1.5-3.0 cm of
  label noise, larger than the grasp tolerance the whole task turns on;
* **all 597 branch episodes** -- their base policy is deliberately corrupted, so their
  ``collector__base_action`` is the corrupted script rather than an expert;
* **post-displacement frames** -- the collector's targets come once from ``initial_truth``, so
  after a perturbation knocks the apple the script is servoing to where the apple *was*.

Surviving roots are asserted against the protocol's frozen 137 / 15 / 8 table. The test split is
never decoded; ``world_model_v2.load_split`` refuses anything but train and val.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from embodied_jepa import readout_labels
from embodied_jepa.contracts import ContractError
from embodied_jepa.policy import (
    FREE_ACTION_INDICES,
    FREE_ACTION_NAMES,
    ClonedPolicy,
    FrozenEncoder,
    NoEncoder,
    state_scale_from_moments,
)
from embodied_jepa.training import BudgetReached, RunClock, peak_rss_bytes
from embodied_jepa.world_model_v2 import (
    PHASE_CLOSE,
    _environment,
    _open,
    _write_json,
    images,
    load_split,
)

PROTOCOL = "apple_policy_v1"
TASK = "TASK-056"

#: scripts/collect_apple_wide.py: FROZEN_SEEDS, AIM_OFFSET_EVERY.
FROZEN_SEEDS = tuple(range(48000, 48200))
AIM_OFFSET_EVERY = 5
AIM_OFFSET_SEEDS = frozenset(
    s for i, s in enumerate(FROZEN_SEEDS) if i % AIM_OFFSET_EVERY == AIM_OFFSET_EVERY - 1
)
#: Protocol §2.2, derived from the committed collection rule. Asserted, not assumed.
SURVIVING_ROOTS = {"train": 137, "val": 15}
#: A frame is dropped once the apple has moved this far from where the script aimed.
DISPLACEMENT_LIMIT_M = 0.01


def is_surviving_root(episode_id: str) -> bool:
    """True only for non-branch, non-aim-offset root episodes."""
    parts = episode_id.split("-")
    if len(parts) != 2:
        return False
    try:
        seed = int(parts[1])
    except ValueError as error:
        raise ContractError(f"unrecognized episode id {episode_id!r}") from error
    return seed in set(FROZEN_SEEDS) and seed not in AIM_OFFSET_SEEDS


def base_actions(store, arrays, *, acknowledge: bool) -> np.ndarray:
    """``collector__base_action`` aligned with ``arrays``' contiguous observation rows.

    The collector records one command per transition (T rows for T+1 observations). The final
    observation of an episode has no command and is left at zero; it is excluded by the sample
    mask, never sampled, and asserted so below.
    """
    if acknowledge is not True:
        raise ContractError("BC targets are privileged collector labels; acknowledge that use")
    total = len(arrays.states)
    result = np.zeros((total, 14), np.float32)
    has_command = np.zeros(total, bool)
    for index, episode_id in enumerate(arrays.episode_ids):
        row = arrays.rows[episode_id]
        labels = readout_labels.load_privileged(
            store.root, row, acknowledge_privileged_training_labels=True
        )
        base = np.asarray(labels["collector__base_action"], np.float32)
        start, length = int(arrays.offsets[index]), int(arrays.lengths[index])
        if not 0 < len(base) <= length:
            raise ContractError(f"base actions of {episode_id} disagree with its observations")
        result[start : start + len(base)] = base
        has_command[start : start + len(base)] = True
    return result, has_command


def sample_mask(store, arrays, *, acknowledge: bool) -> tuple[np.ndarray, dict]:
    """Rows eligible as BC training samples, plus the accounting a reviewer can check."""
    targets = arrays.targets
    total = len(arrays.states)
    _, has_command = base_actions(store, arrays, acknowledge=acknowledge)

    dropped = targets["apple_dropped"][:, 0] > 0.5
    # Post-displacement: the apple has moved from where the script aimed, so the recorded
    # command is servoing to a stale target. Measured per episode against its own first frame.
    displaced = np.zeros(total, bool)
    for index in range(len(arrays.episode_ids)):
        start, length = int(arrays.offsets[index]), int(arrays.lengths[index])
        span = slice(start, start + length)
        apple = targets["apple_position"][span]
        drift = np.linalg.norm(apple - apple[0], axis=1)
        phase = arrays.phase[span]
        # Only before the grasp: after a successful close the apple is *meant* to move.
        pre_grasp = phase < PHASE_CLOSE
        displaced[span] = (drift > DISPLACEMENT_LIMIT_M) & pre_grasp

    keep = has_command & ~dropped & ~displaced
    accounting = {
        "observation_rows": int(total),
        "rows_without_a_command": int((~has_command).sum()),
        "rows_apple_dropped": int(dropped.sum()),
        "rows_post_displacement_pre_grasp": int(displaced.sum()),
        "rows_kept": int(keep.sum()),
        "displacement_limit_m": DISPLACEMENT_LIMIT_M,
    }
    return keep, accounting


def load_bc_split(store, split, cameras, *, workers, limit, acknowledge):
    """Decode a split, keep only surviving roots, and assert the frozen surviving count."""
    arrays = load_split(
        store,
        split,
        cameras,
        workers=workers,
        limit=limit,
        acknowledge_privileged_training_labels=acknowledge,
    )
    surviving = [e for e in arrays.episode_ids if is_surviving_root(e)]
    if limit is None and len(surviving) != SURVIVING_ROOTS[split]:
        raise ContractError(
            f"{split}: expected {SURVIVING_ROOTS[split]} surviving roots, found "
            f"{len(surviving)}; the cohort does not match the protocol's exclusion table"
        )
    keep, accounting = sample_mask(store, arrays, acknowledge=acknowledge)
    root_rows = np.zeros(len(arrays.states), bool)
    for index, episode_id in enumerate(arrays.episode_ids):
        if is_surviving_root(episode_id):
            start = int(arrays.offsets[index])
            root_rows[start : start + int(arrays.lengths[index])] = True
    accounting["rows_in_excluded_episodes"] = int((~root_rows).sum())
    accounting["surviving_root_episodes"] = len(surviving)
    accounting["episodes_decoded"] = len(arrays.episode_ids)
    rows = np.flatnonzero(keep & root_rows)
    accounting["rows_sampled"] = int(len(rows))
    return arrays, rows, accounting


def precompute_features(source, arrays, rows, *, chunk=256):
    """Frozen image features for every sampled row, computed once and reused by every step."""
    if not source.feature_dim:
        return None
    out = np.empty((len(rows), source.feature_dim), np.float32)
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        out[start : start + len(part)] = (
            source.features(images(arrays, part)).detach().cpu().numpy()
        )
    return out


def action_error(predicted, target) -> dict:
    """Per-dimension median absolute error in normalized action units."""
    error = np.abs(predicted - target)
    per_dimension = {
        name: float(np.median(error[:, i])) for i, name in enumerate(FREE_ACTION_NAMES)
    }
    arm = error[:, :6]
    return {
        "median_abs_error": float(np.median(error)),
        "median_abs_error_arm": float(np.median(arm)),
        "median_abs_error_grasp": float(np.median(error[:, 6])),
        "per_dimension": per_dimension,
        "samples": int(len(error)),
    }


def evaluate_policy(policy, arrays, rows, features, targets, *, chunk=512) -> dict:
    """Held-out action error plus the eligibility rule's collapse check."""
    import torch

    policy.eval()
    predictions = np.empty((len(rows), len(FREE_ACTION_INDICES)), np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            visual = None
            if features is not None:
                visual = torch.from_numpy(features[start : start + len(part)]).to(
                    policy.device_name
                )
            state = policy.normalized_state(arrays.states[part], arrays.mask[part])
            predictions[start : start + len(part)] = policy(visual, state).detach().cpu().numpy()
    result = action_error(predictions, targets)
    std = predictions.std(axis=0)
    result["output_std_min"] = float(std.min())
    result["output_std_per_dimension"] = {
        name: float(std[i]) for i, name in enumerate(FREE_ACTION_NAMES)
    }
    return result


def selectable(decision, best, eligibility) -> bool:
    """Best-by-val, but a collapsed head is never selected however good its error looks."""
    if decision["output_std_min"] < eligibility["min_output_std"]:
        return False
    return best is None or decision["median_abs_error"] < best[1]


def train(
    config_path,
    *,
    arm,
    encoder,
    output,
    steps,
    batch_size,
    seed,
    device,
    max_seconds,
    validation_every,
    eligibility,
    workers=8,
    limit_episodes=None,
    acknowledge_privileged_training_labels=False,
):
    """Fixed-budget behaviour cloning. ``output`` is best-by-val; ``.latest.pt`` is the last."""
    import torch

    from embodied_jepa.config import MODELS
    from embodied_jepa.training import source_identity

    if acknowledge_privileged_training_labels is not True:
        raise ContractError("BC targets are privileged collector labels; acknowledge that use")
    if encoder not in ("none", "frozen", "random", "finetune"):
        raise ContractError("encoder must be none, frozen, random or finetune")
    config, store, settings, cameras = _open(config_path)
    output = Path(output)
    paths = {
        "best": output,
        "latest": output.with_name(output.stem + ".latest.pt"),
        "report": output.with_name(output.stem + ".run.json"),
        "curves": output.with_name(output.stem + ".metrics.jsonl"),
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite existing policy training artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    clock = RunClock()
    report = {
        "format_version": 1,
        "protocol": PROTOCOL,
        "task": TASK,
        "arm": arm,
        "encoder": encoder,
        "status": "running",
        "config": str(Path(config_path).resolve()),
        "backend": config.backend,
        "cameras": list(cameras),
        "device": device,
        "seed": seed,
        "budget": {
            "steps": steps,
            "batch_size": batch_size,
            "max_seconds": max_seconds,
            "validation_every": validation_every,
        },
        "target": "collector__base_action (free components only)",
        "free_action_names": list(FREE_ACTION_NAMES),
        "privileged_label_use": "BC targets and the training mask only; never a model input",
        "source": source_identity(),
        "environment": _environment(),
        "test_episodes_decoded": 0,
        "validation": [],
    }
    _write_json(paths["report"], report)

    policy, completed, best, failed = None, 0, None, None
    try:
        train_arrays, train_rows, train_counts = load_bc_split(
            store, "train", cameras, workers=workers, limit=limit_episodes, acknowledge=True
        )
        val_arrays, val_rows, val_counts = load_bc_split(
            store, "val", cameras, workers=workers, limit=limit_episodes, acknowledge=True
        )
        report["data"] = {"train": train_counts, "val": val_counts}
        clock.check(max_seconds)

        source = NoEncoder()
        if encoder != "none":
            model = MODELS.create(
                config.backend,
                state_schema=store.state_schema,
                device=device,
                seed=seed,
                config=settings,
            )
            if encoder in ("frozen", "finetune"):
                model.load(config.checkpoint)
            source = FrozenEncoder(model, frozen=encoder != "finetune")
        report["feature_source"] = source.provenance()

        normalization = store.manifest["normalization"]
        if normalization["fit_split"] != "train" or list(normalization["episode_ids"]) != list(
            store.manifest["splits"]["train"]
        ):
            raise ContractError("state normalization must be fitted on the train split only")
        moments = normalization["stats"]["observation.state"]
        mean, scale = state_scale_from_moments(moments["mean"], moments["std"])
        policy = ClonedPolicy(
            store.state_schema,
            source,
            device=device,
            seed=seed,
            metadata={"protocol": PROTOCOL, "arm": arm},
        )
        policy.fit_state_normalization(
            mean, scale, training_episode_ids=normalization["episode_ids"]
        )
        report["policy_parameters"] = int(sum(p.numel() for p in policy.head.parameters()))
        report["policy_implementation_sha256"] = policy.implementation_sha256

        train_base, _ = base_actions(store, train_arrays, acknowledge=True)
        val_base, _ = base_actions(store, val_arrays, acknowledge=True)
        train_targets = train_base[train_rows][:, list(FREE_ACTION_INDICES)]
        val_targets = val_base[val_rows][:, list(FREE_ACTION_INDICES)]
        report["data"]["train"]["target_sha256"] = hashlib.sha256(
            train_targets.tobytes()
        ).hexdigest()

        cached = (
            precompute_features(source, train_arrays, train_rows) if encoder != "finetune" else None
        )
        val_features = precompute_features(source, val_arrays, val_rows)
        report["data"]["features_precomputed"] = cached is not None
        clock.check(max_seconds)

        sampler = np.random.default_rng(seed)
        target_tensor = torch.from_numpy(train_targets).to(device)

        def validate():
            nonlocal best
            decision = evaluate_policy(policy, val_arrays, val_rows, val_features, val_targets)
            improved = selectable(decision, best, eligibility)
            event = {
                "kind": "validation",
                "step": completed,
                "elapsed_seconds": clock.elapsed(),
                "selected": bool(improved),
            } | decision
            report["validation"].append(event)
            with paths["curves"].open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            if improved:
                best = (completed, decision["median_abs_error"])
                policy.metadata["runner_state"] = {"step": completed, "selection": decision}
                if paths["best"].exists():
                    paths["best"].unlink()
                policy.save(paths["best"])
            _write_json(paths["report"], report)

        validate()
        for step in range(1, steps + 1):
            clock.check(max_seconds)
            picked = sampler.integers(len(train_rows), size=batch_size)
            policy.train()
            visual = None
            if cached is not None:
                visual = torch.from_numpy(cached[picked]).to(device)
            elif source.feature_dim:
                visual = source.features(images(train_arrays, train_rows[picked]))
            rows = train_rows[picked]
            state = policy.normalized_state(train_arrays.states[rows], train_arrays.mask[rows])
            predicted = policy(visual, state)
            loss = torch.nn.functional.smooth_l1_loss(predicted, target_tensor[picked])
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in policy.optimizer.param_groups for p in group["params"]],
                policy.config["gradient_clip"],
            )
            policy.optimizer.step()
            policy.updates += 1
            completed = step
            if not np.isfinite(loss.item()):
                raise ContractError("non-finite training loss")
            if step % 50 == 0 or step == 1:
                with paths["curves"].open("a") as stream:
                    stream.write(
                        json.dumps(
                            {"kind": "train", "step": step, "loss": float(loss.item())},
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if step % validation_every == 0 or step == steps:
                validate()
        report["status"] = "completed" if best is not None else "selection_failed"
    except BudgetReached as error:
        report["status"] = str(error)
    except Exception as error:  # noqa: BLE001 - recorded, re-raised below
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        failed = error
    finally:
        if policy is not None:
            try:
                policy.metadata["runner_state"] = {"step": completed, "latest": True}
                if paths["latest"].exists():
                    paths["latest"].unlink()
                policy.save(paths["latest"])
            except Exception as error:  # noqa: BLE001 - must not hide the real failure
                report["latest_checkpoint_error"] = f"{type(error).__name__}: {error}"
        report.update(
            completed_steps=completed,
            best_step=None if best is None else best[0],
            best_selection_score=None if best is None else best[1],
            **clock.snapshot(),
            peak_host_rss_bytes=peak_rss_bytes(),
        )
        _write_json(paths["report"], report)
    if failed is not None:
        raise failed
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--encoder", required=True, choices=("none", "frozen", "random", "finetune")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--validation-every", type=int, default=2000)
    parser.add_argument("--min-output-std", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int, help="smoke subsets only")
    parser.add_argument("--acknowledge-privileged-training-labels", action="store_true")
    args = parser.parse_args()
    report = train(
        args.config,
        arm=args.arm,
        encoder=args.encoder,
        output=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        max_seconds=args.max_seconds,
        validation_every=args.validation_every,
        eligibility={"min_output_std": args.min_output_std},
        workers=args.workers,
        limit_episodes=args.limit_episodes,
        acknowledge_privileged_training_labels=args.acknowledge_privileged_training_labels,
    )
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in ("status", "completed_steps", "best_step", "best_selection_score")
            },
            indent=2,
        )
    )
    return 0 if report.get("status") == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
