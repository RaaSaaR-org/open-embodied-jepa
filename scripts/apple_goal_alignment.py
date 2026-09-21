"""Bounded offline goal-image alignment; preserves the frozen world model and TEST split."""

from __future__ import annotations

import time

ENTRY = time.time(), time.monotonic()
# ruff: noqa: E402
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

FIELDS = tuple(
    f"{name}_joint.position"
    for name in (
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
    )
)
PHASES = ("orient", "descend", "close", "lift", "transfer", "release_high")
BUDGETS = {"prepare": 120.0, "fit": 300.0, "evaluate": 60.0}
DATA_SHA = "6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331"
CHECKPOINT_SHA = "0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5"
PROTOCOL = "docs/experiments/apple_goal_alignment_v1.md"
ROOT = Path(__file__).resolve().parents[1]


def elapsed(entry=ENTRY):
    return max(0.0, time.time() - entry[0], time.monotonic() - entry[1])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def resolve_fields(schema):
    names = list(schema.names)
    indices = tuple(names.index(name) for name in FIELDS)
    if len(set(indices)) != 7 or any(schema.units[i] != "rad" for i in indices):
        raise ValueError("seven named right-arm fields must be distinct radian positions")
    return indices


def sample_table(manifest):
    """Metadata only. No success filtering, TEST decoding or parent repartition."""
    rows = {r["episode_id"]: r for r in manifest["episodes"]}
    owners = {episode: split for split, ids in manifest["splits"].items() for episode in ids}
    if len(owners) != sum(map(len, manifest["splits"].values())) or set(owners) != set(rows):
        raise ValueError("split membership must cover each episode exactly once")
    samples = []
    missing = []
    extras = []
    for split in ("train", "val"):
        for episode in sorted(manifest["splits"][split]):
            row = rows[episode]
            meta = row["metadata"]
            parent = meta.get("parent_episode_id", episode)
            if owners.get(parent) != split or row["session_id"] != rows[parent]["session_id"]:
                raise ValueError("parent/session crosses frozen split")
            labels = meta["phase_labels"]
            if len(labels) != row["length"] - 1:
                raise ValueError("phase labels must align with transitions")
            branch = parent != episode
            if branch:
                phase = labels[0]
                if phase not in PHASES or any(p != phase for p in labels) or row["length"] != 17:
                    raise ValueError("branches require one declared phase and complete H16")
                selected = [(phase, 8), (phase, 16)]
            else:
                selected = []
                extras += [
                    dict(episode_id=episode, phase=p) for p in sorted(set(labels) - set(PHASES))
                ]
                for phase in PHASES:
                    indices = [i for i, p in enumerate(labels) if p == phase]
                    if len(indices) < 4:
                        missing.append(
                            dict(episode_id=episode, phase=phase, available=len(indices))
                        )
                        continue
                    selected += [(phase, indices[(len(indices) - 1) * k // 3]) for k in range(4)]
            for phase, frame in selected:
                samples.append(
                    dict(
                        split=split,
                        parent=parent,
                        episode_id=episode,
                        frame=frame,
                        phase=phase,
                        source="branch" if branch else "original",
                        root_id=meta.get("root_id"),
                        branch=meta.get("branch"),
                    )
                )
    samples.sort(key=lambda r: (0 if r["split"] == "train" else 1, r["episode_id"], r["frame"]))
    identities = [(r["episode_id"], r["frame"]) for r in samples]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate selected frame")
    return dict(
        samples=samples,
        missing=missing,
        unsampled_extra_phases=extras,
        test_images_decoded=False,
        sampling_sha256=json_hash(samples),
    )


def frame_weights(rows):
    """Uniform parents, uniform available source strata, uniform selected frames."""
    import numpy as np

    parents = sorted({r["parent"] for r in rows})
    strata = {p: sorted({r["source"] for r in rows if r["parent"] == p}) for p in parents}
    counts = {
        (p, s): sum(r["parent"] == p and r["source"] == s for r in rows)
        for p in parents
        for s in strata[p]
    }
    return np.array(
        [
            1 / len(parents) / len(strata[r["parent"]]) / counts[r["parent"], r["source"]]
            for r in rows
        ],
        dtype=np.float64,
    )


def head_factory():
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(1728, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 14),
    )


def estimate_neural(head, features):
    import torch

    with torch.no_grad():
        outputs = head(torch.as_tensor(features, dtype=torch.float32))
        return outputs[:, :7].numpy(), outputs[:, 7:].clamp(-6, 3).exp().numpy()


def nearest(features, reference, labels, *, chunk=128):
    import numpy as np

    result = []
    variances = []
    ids = []
    # Reference order is frozen lexicographic episode/frame order. Stable sorting
    # gives an explicit tie rule, with the same exact distance for every route.
    for start in range(0, len(features), chunk):
        x = features[start : start + chunk].astype(np.float64)
        ref = reference.astype(np.float64)
        distances = np.maximum(
            (x * x).sum(1)[:, None] + (ref * ref).sum(1)[None] - 2 * x @ ref.T, 0
        )
        order = np.argsort(distances, axis=1, kind="stable")[:, : min(8, len(ref))]
        ids.extend(order[:, 0].tolist())
        result.extend(labels[order[:, 0]])
        variances.extend(labels[order].var(1))
    return np.asarray(result), np.asarray(variances), ids


def identities(args):
    files = {
        str(p.relative_to(ROOT)): p for p in sorted((ROOT / "src/embodied_jepa").rglob("*.py"))
    }
    files.update(
        script=Path(__file__),
        protocol=ROOT / PROTOCOL,
        checkpoint=args.checkpoint,
        dataset=args.dataset / "meta/jepa_manifest.json",
        checkpoint_loader=ROOT / "scripts/evaluate_apple.py",
    )
    for stage in ("prepare", "fit"):
        if list(BUDGETS).index(stage) >= list(BUDGETS).index(args.stage):
            break
        for path in sorted((args.output / stage).glob("*")):
            if path.is_file() and path.suffix in (".json", ".npz", ".pt"):
                files[f"{stage}/{path.name}"] = path
    result = {k: digest(p) for k, p in files.items()}
    if result["dataset"] != DATA_SHA or result["checkpoint"] != CHECKPOINT_SHA:
        raise ValueError("preregistered dataset/checkpoint hashes differ")
    result["revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return result


def load_model(args):
    spec = importlib.util.spec_from_file_location(
        "goal_checkpoint_loader", ROOT / "scripts/evaluate_apple.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    return loader.checkpoint_model(args.dataset, args.checkpoint)


def save_npz(path, **values):
    import numpy as np

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(path)


def prepare(args, check, report):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    store, model = load_model(args)
    indices = resolve_fields(store.state_schema)
    table = sample_table(store.manifest)
    folder = args.output / "prepare"
    write(folder / "samples.json", table)
    if table["missing"] or [
        sum(r["split"] == s for r in table["samples"]) for s in ("train", "val")
    ] != [1392, 360]:
        raise ValueError("declared sample coverage is incomplete")
    samples = table["samples"]
    features = np.empty((len(samples), 1728), np.float32)
    targets = np.empty((len(samples), 7), np.float32)
    roots = {}
    report["decoded_episode_ids"] = []
    train_rows = [r for r in samples if r["split"] == "train"]
    train_count = len(train_rows)
    train_ids = sorted({r["episode_id"] for r in train_rows})
    val_ids = sorted({r["episode_id"] for r in samples if r["split"] == "val"})
    bank_frozen = False
    for episode_id in train_ids + val_ids:
        if episode_id in val_ids and not bank_frozen:
            weights = frame_weights(train_rows)
            save_npz(
                folder / "train_features.npz",
                features=features[:train_count],
                targets=targets[:train_count],
            )
            save_npz(
                folder / "nn_bank.npz",
                features=features[:train_count],
                targets=targets[:train_count],
                weights=weights,
                mean=(targets[:train_count] * weights[:, None]).sum(0),
                variance=(
                    (targets[:train_count] - (targets[:train_count] * weights[:, None]).sum(0)) ** 2
                    * weights[:, None]
                ).sum(0),
            )
            write(folder / "train_samples.json", train_rows)
            report["nn_bank_sha256_before_val"] = digest(folder / "nn_bank.npz")
            report["bank_frozen_before_val_decode"] = True
            write(folder / "report.json", report)
            bank_frozen = True

        check()
        report["decode_in_progress"] = episode_id
        write(folder / "report.json", report)
        episode = store.read_episode(episode_id)
        report["decoded_episode_ids"].append(episode_id)
        report["decode_in_progress"] = None
        positions = [i for i, r in enumerate(samples) if r["episode_id"] == episode_id]
        frames = [samples[i]["frame"] for i in positions]
        if not episode.state_mask[frames][:, indices].all():
            raise ValueError("unobserved target joint")
        images = {model.config["camera"]: episode.observations[model.config["camera"]][frames]}
        features[positions] = model.encode_goal(images).values.cpu().numpy()
        mean = model.sensor_mean[model.visual_dimension + np.array(indices)].numpy()
        scale = model.sensor_scale[model.visual_dimension + np.array(indices)].numpy()
        targets[positions] = (episode.robot_states[frames][:, indices] - mean) / scale
        first = samples[positions[0]]
        if first["split"] == "val" and first["source"] == "branch":
            root = roots.setdefault(
                first["root_id"], dict(parent=first["parent"], phase=first["phase"], branches=[])
            )
            rgb = episode.observations[model.config["camera"]][0]
            state = episode.robot_states[0]
            mask = episode.state_mask[0]
            if "rgb" in root and not all(
                np.array_equal(root[k], v)
                for k, v in [
                    ("rgb", rgb),
                    ("state", state),
                    ("mask", mask),
                    ("timestamps", episode.timestamps),
                ]
            ):
                raise ValueError("sibling starting observations differ")
            root.update(rgb=rgb, state=state, mask=mask, timestamps=episode.timestamps.copy())
            root["branches"].append(
                dict(
                    name=first["branch"],
                    episode_id=episode_id,
                    actions=episode.actions,
                    sample8=positions[0] - train_count,
                    sample16=positions[1] - train_count,
                )
            )
    ordered = sorted(roots)
    rootrows = []
    actions = []
    rgb = []
    states = []
    masks = []
    times = []
    endpoints = []
    for root_id in ordered:
        root = roots[root_id]
        branches = sorted(root["branches"], key=lambda r: r["name"])
        if len(branches) != 8 or len({r["name"] for r in branches}) != 8:
            raise ValueError("eight siblings required")
        rootrows.append(
            dict(
                root_id=root_id,
                parent=root["parent"],
                phase=root["phase"],
                branches=[
                    {k: r[k] for k in ("name", "episode_id", "sample8", "sample16")}
                    for r in branches
                ],
            )
        )
        actions.append(np.stack([r["actions"] for r in branches]))
        rgb.append(root["rgb"])
        states.append(root["state"])
        masks.append(root["mask"])
        times.append(root["timestamps"])
        endpoints.append([[r["sample8"], r["sample16"]] for r in branches])
    check()
    save_npz(
        folder / "val_features.npz",
        features=features[train_count:],
        targets=targets[train_count:],
        q_mean=mean,
        q_scale=scale,
        field_indices=np.asarray(indices),
        root_rgb=np.stack(rgb),
        root_states=np.stack(states),
        root_masks=np.stack(masks),
        root_actions=np.stack(actions),
        root_timestamps=np.stack(times),
        root_endpoints=np.asarray(endpoints),
    )
    write(folder / "roots.json", rootrows)
    write(folder / "val_samples.json", samples[train_count:])
    report.update(
        samples=len(samples),
        counts={s: sum(r["split"] == s for r in samples) for s in ("train", "val")},
        sample_sha256=table["sampling_sha256"],
        missing=table["missing"],
        unsampled_extra_phases=table["unsampled_extra_phases"],
    )


def fit(args, check, report):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    rows = json.loads((args.output / "prepare/train_samples.json").read_text())
    if any(r["split"] != "train" for r in rows):
        raise ValueError("fit received nonTRAIN row")
    ids = list(range(len(rows)))
    with np.load(args.output / "prepare/train_features.npz", allow_pickle=False) as cache:
        x = cache["features"].copy()
        y = cache["targets"].copy()
    weights = frame_weights(rows)
    rng = np.random.default_rng(0)
    head = head_factory()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    folder = args.output / "fit"
    report["updates"] = 0
    try:
        for step in range(2000):
            check()
            selected = rng.choice(len(ids), size=128, replace=True, p=weights)
            values = head(torch.from_numpy(x[selected]))
            mu, logvar = values[:, :7], values[:, 7:].clamp(-6, 3)
            loss = (
                0.5
                * (
                    logvar + (torch.from_numpy(y[selected]) - mu).square() * torch.exp(-logvar)
                ).mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite head loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            report.update(updates=step + 1, last_loss=float(loss.detach()))
            if (step + 1) % 100 == 0:
                write(folder / "report.json", report)
                with (folder / "curves.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=step + 1, loss=report["last_loss"])) + "\n")
    finally:
        torch.save(
            dict(
                head=head.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                updates=report["updates"],
                train_rows=ids,
                sample_sha256=json_hash(rows),
                fields=FIELDS,
                architecture=[1728, 128, 64, 14],
                config=dict(
                    seed=0,
                    batch_size=128,
                    updates=2000,
                    lr=1e-3,
                    weight_decay=1e-4,
                    gradient_clip=10,
                    logvar_bounds=[-6, 3],
                ),
            ),
            folder / "head.pt",
        )


def encoder_errors(rows, targets, estimates, variances, q_scale):
    import numpy as np

    weights = frame_weights(rows)
    parents = sorted({r["parent"] for r in rows})
    result = {}
    for name, values in estimates.items():
        error = np.square(values - targets)
        parent_errors = {}
        for parent in parents:
            ids = [i for i, r in enumerate(rows) if r["parent"] == parent]
            parent_errors[parent] = float(np.average(error[ids].mean(1), weights=weights[ids]))
        item = dict(
            mse=float(np.average(error.mean(1), weights=weights)),
            parents=parent_errors,
            joint_rmse_rad=np.sqrt(np.average(error, axis=0, weights=weights)) * q_scale,
            record_mse=error.mean(1),
        )
        if name in variances:
            variance = variances[name]
            spread = np.sqrt(variance.mean(1))
            risk = {}
            for coverage in (1.0, 0.75, 0.5):
                parent_risk = {}
                retained = {}
                for parent in parents:
                    ids = [i for i, r in enumerate(rows) if r["parent"] == parent]
                    ordered = sorted(
                        ids,
                        key=lambda i: (float(spread[i]), rows[i]["episode_id"], rows[i]["frame"]),
                    )
                    selected = ordered[: max(1, int(np.ceil(coverage * len(ordered))))]
                    parent_risk[parent] = float(
                        np.average(error[selected].mean(1), weights=weights[selected])
                    )
                    retained[parent] = dict(
                        count=len(selected),
                        total_count=len(ids),
                        weighted_mass=float(weights[selected].sum() / weights[ids].sum()),
                    )
                risk[str(coverage)] = dict(
                    mse=float(np.mean(list(parent_risk.values()))),
                    parents=parent_risk,
                    nominal_count_fraction=coverage,
                    retained_by_parent=retained,
                )
            item.update(risk_coverage=risk, uncertainty_rms=spread)
            if name == "neural":
                covered = np.abs(values - targets) <= 1.644854 * np.sqrt(variance)
                item["marginal_90_coverage"] = np.average(covered, axis=0, weights=weights)
                item["marginal_90_width_normalized"] = np.average(
                    2 * 1.644854 * np.sqrt(variance), axis=0, weights=weights
                )
                item["coverage_by_parent"] = {
                    p: np.average(
                        covered[[i for i, r in enumerate(rows) if r["parent"] == p]],
                        axis=0,
                        weights=weights[[i for i, r in enumerate(rows) if r["parent"] == p]],
                    )
                    for p in parents
                }
        result[name] = item
    return result


def ranking_pairs(measured, goal, costs):
    import numpy as np

    labels = np.square(measured - goal).mean(1)
    pairs = []
    for i in range(len(measured)):
        for j in range(i + 1, len(measured)):
            separation = float(np.sqrt(np.square(measured[i] - measured[j]).mean()))
            margin = 1e-6 + 1e-4 * max(1.0, float(labels[i]), float(labels[j]))
            eligible = separation >= 0.1 and abs(labels[i] - labels[j]) > margin
            row = dict(
                i=i,
                j=j,
                arm_rms=separation,
                label_i=float(labels[i]),
                label_j=float(labels[j]),
                eligible=bool(eligible),
                exclusion=None
                if eligible
                else ("low_arm_separation" if separation < 0.1 else "label_tie"),
                scores={},
            )
            if eligible:
                for name, values in costs.items():
                    a, b = float(values[i]), float(values[j])
                    tie = abs(a - b) <= 1e-10 + 1e-6 * max(abs(a), abs(b))
                    row["scores"][name] = 0.5 if tie else float((a < b) == (labels[i] < labels[j]))
            pairs.append(row)
    return pairs


def aggregate_rankings(cases, parents):
    import numpy as np

    roots = {}
    coverage = []
    for case in cases:
        eligible = [p for p in case["pairs"] if p["eligible"]]
        coverage.append(
            dict(
                root_id=case["root_id"],
                goal=case["goal"],
                eligible=len(eligible),
                total=len(case["pairs"]),
            )
        )
        if not eligible:
            continue
        means = {
            name: float(np.mean([p["scores"][name] for p in eligible]))
            for name in eligible[0]["scores"]
        }
        root = roots.setdefault(case["root_id"], dict(parent=case["parent"], goals=[]))
        root["goals"].append(means)
    summaries = {}
    counts = {p: 0 for p in parents}
    for root in roots.values():
        counts[root["parent"]] += 1
        for name in root["goals"][0]:
            summaries.setdefault(name, {}).setdefault(root["parent"], []).append(
                float(np.mean([g[name] for g in root["goals"]]))
            )
    parent_scores = {
        name: {p: float(np.mean(v)) for p, v in values.items()}
        for name, values in summaries.items()
    }
    return dict(
        scores={
            name: float(np.mean(list(values.values()))) for name, values in parent_scores.items()
        },
        parents=parent_scores,
        informative_roots=counts,
        coverage=coverage,
        sufficient_coverage=all(counts[p] >= 2 for p in parents) and len(parents) == 3,
    )


def route_decisions(errors, ranking, *, complete):
    import numpy as np

    result = {}
    parents = sorted(errors["mean"]["parents"])
    for name in ("nn", "neural"):
        if name not in errors:
            result[name] = dict(status="unverified", reason="candidate_incomplete")
            continue
        baseline = errors["mean"]["mse"]
        r = None if baseline <= 0 else 1 - errors[name]["mse"] / baseline
        key = name + "_pred"
        pixel = ranking["parents"].get("pixel_pred", {})
        method = ranking["parents"].get(key, {})
        available = (
            complete
            and ranking["sufficient_coverage"]
            and r is not None
            and set(method) == set(parents)
            and set(pixel) == set(parents)
        )
        deltas = {p: method[p] - pixel[p] for p in parents} if available else {}
        delta = float(np.mean(list(deltas.values()))) if deltas else None
        passed = available and r >= 0.2 and delta >= 0.05 and all(v > 0 for v in deltas.values())
        result[name] = dict(
            status=("passed" if passed else "failed") if available else "unverified",
            error_reduction=r,
            ranking_delta=delta,
            parent_deltas=deltas,
        )
        if available:
            random = np.random.default_rng(0)
            draw = random.integers(0, 3, size=(2000, 3))
            e = np.array([errors[name]["parents"][p] for p in parents])
            b = np.array([errors["mean"]["parents"][p] for p in parents])
            a = np.array([method[p] for p in parents])
            d = np.array([deltas[p] for p in parents])
            denominator = b[draw].mean(1)
            invalid_draws = int((denominator <= 0).sum())
            result[name]["error_reduction_bootstrap_invalid_draws"] = invalid_draws
            result[name]["error_reduction_ci95"] = (
                None
                if invalid_draws
                else np.quantile(1 - e[draw].mean(1) / denominator, [0.025, 0.975]).tolist()
            )
            for label, values in [
                ("ranking", a[draw].mean(1)),
                ("delta", d[draw].mean(1)),
            ]:
                result[name][label + "_ci95"] = np.quantile(values, [0.025, 0.975]).tolist()
    result["neural_preferred_replacement"] = bool(
        result["neural"]["status"] == "passed"
        and "neural" in errors
        and errors["neural"]["mse"] < errors["nn"]["mse"]
        and ranking["scores"]["neural_pred"] > ranking["scores"]["nn_pred"]
    )
    return result


def json_arrays(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_arrays(v) for v in value]
    return value


def evaluate(args, check, report):
    import numpy as np
    import torch

    from embodied_jepa.contracts import RobotState

    torch.set_num_threads(4)
    folder = args.output / "evaluate"
    prep = args.output / "prepare"
    rows = json.loads((prep / "val_samples.json").read_text())
    roots = json.loads((prep / "roots.json").read_text())
    if any(r["split"] != "val" for r in rows):
        raise ValueError("evaluation cache is not exclusively VAL")
    with np.load(prep / "val_features.npz", allow_pickle=False) as file:
        cache = {k: file[k].copy() for k in file.files}
    with np.load(prep / "nn_bank.npz", allow_pickle=False) as file:
        bank = {k: file[k].copy() for k in file.files}
    check()
    x, y = cache["features"], cache["targets"]
    mean = np.broadcast_to(bank["mean"], y.shape).copy()
    nn, nn_var, neighbor_ids = nearest(x, bank["features"], bank["targets"])
    estimates = {"mean": mean, "nn": nn}
    variances = {"nn": nn_var}
    if report.get("fit_validation", {}).get("eligible", False):
        try:
            checkpoint = torch.load(
                args.output / "fit/head.pt", map_location="cpu", weights_only=True
            )
            if checkpoint["updates"] != 2000:
                raise ValueError("completed head lacks fixed updates")
            head = head_factory()
            head.load_state_dict(checkpoint["head"])
            head.eval()
            candidate, uncertainty = estimate_neural(head, x)
            if not np.isfinite(candidate).all() or not np.isfinite(uncertainty).all():
                raise ValueError("nonfinite neural inference")
            estimates["neural"], variances["neural"] = candidate, uncertainty
        except Exception as error:
            report["neural_error"] = f"{type(error).__name__}: {error}"
    errors = encoder_errors(rows, y, estimates, variances, cache["q_scale"])
    write(folder / "encoder_errors.json", json_arrays(errors))
    write(
        folder / "goal_estimates.json",
        json_arrays(dict(means=estimates, variances=variances, nn_reference_indices=neighbor_ids)),
    )
    store, model = load_model(args)
    indices = np.array(resolve_fields(store.state_schema)) + model.visual_dimension
    parents = sorted({r["parent"] for r in rows})
    cases = {8: [], 16: []}
    matched = {8: [], 16: []}
    ambiguities = []
    report["roots"] = [dict(root_id=r["root_id"], status="not_started") for r in roots]
    write(folder / "report.json", report)
    for root_index, root in enumerate(roots):
        check()
        state = RobotState(
            cache["root_states"][root_index : root_index + 1],
            cache["root_masks"][root_index : root_index + 1],
            cache["root_timestamps"][root_index : root_index + 1, 0],
            store.state_schema,
        )
        images = {model.config["camera"]: cache["root_rgb"][root_index : root_index + 1]}
        latent = model.encode(images, state)
        prediction = (
            model.predict(latent, cache["root_actions"][root_index : root_index + 1])
            .values[0]
            .cpu()
            .numpy()
        )
        save_npz(
            folder / f"prediction-{root_index:02d}.npz",
            sensor_prediction=prediction,
            q_indices=indices,
            visual_mean=model.sensor_mean[:1728].numpy(),
            visual_scale=model.sensor_scale[:1728].numpy(),
        )
        for horizon in (8, 16):
            endpoint = cache["root_endpoints"][root_index, :, 0 if horizon == 8 else 1]
            measured = y[endpoint]
            visual = x[endpoint]
            predq = prediction[:, horizon - 1, indices]
            predvisual = prediction[:, horizon - 1, :1728]
            goals = [
                i
                for i, r in enumerate(rows)
                if r["source"] == "original"
                and r["phase"] == root["phase"]
                and r["parent"] != root["parent"]
            ]
            if len(goals) != 8:
                raise ValueError("cross-parent cohort must supply exactly eight goals")
            for cohort, goal_ids in ((cases[horizon], goals), (matched[horizon], endpoint)):
                for goal in goal_ids:
                    costs = {
                        "pixel_pred": np.square(predvisual - x[goal]).mean(1),
                        "pixel_measured": np.square(visual - x[goal]).mean(1),
                    }
                    for name, values in estimates.items():
                        if name == "mean":
                            continue
                        costs[name + "_pred"] = np.square(predq - values[goal]).mean(1)
                        costs[name + "_measured"] = np.square(measured - values[goal]).mean(1)
                    cohort.append(
                        dict(
                            root_id=root["root_id"],
                            parent=root["parent"],
                            phase=root["phase"],
                            goal=dict(
                                episode_id=rows[goal]["episode_id"], frame=rows[goal]["frame"]
                            ),
                            costs=json_arrays(costs),
                            pairs=ranking_pairs(measured, y[goal], costs),
                        )
                    )
            if horizon == 16:
                raw_visual = (
                    visual * model.sensor_scale[:1728].numpy() + model.sensor_mean[:1728].numpy()
                )
                for i in range(8):
                    for j in range(i + 1, 8):
                        q_rms = float(np.sqrt(np.square(measured[i] - measured[j]).mean()))
                        rgb_rms = float(np.sqrt(np.square(raw_visual[i] - raw_visual[j]).mean()))
                        ambiguities.append(
                            dict(
                                root_id=root["root_id"],
                                i=i,
                                j=j,
                                arm_rms=q_rms,
                                rgb_rms=rgb_rms,
                                ambiguous=q_rms >= 0.1 and rgb_rms < 1 / 255,
                            )
                        )
        report["roots"][root_index]["status"] = "completed"
        write(folder / "report.json", report)
    summaries = {str(h): aggregate_rankings(cases[h], parents) for h in (8, 16)}
    decisions = route_decisions(
        errors, summaries["16"], complete=len(rows) == 360 and len(bank["targets"]) == 1392
    )
    write(folder / "primary_cases.json", json_arrays(cases))
    write(folder / "matched_secondary.json", json_arrays(matched))
    write(folder / "ambiguities.json", ambiguities)
    report.update(
        rankings=summaries,
        decisions=decisions,
        ambiguity_examples=sum(r["ambiguous"] for r in ambiguities),
        neural_available="neural" in estimates,
        nn_available=True,
        test_images_decoded=False,
    )


def seal_stage(folder):
    artifacts = {
        p.name: digest(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name not in ("seal.json", "worker.log")
    }
    write(folder / "seal.json", dict(artifacts=artifacts))


def verify_producer(args, stage, current, *, require_complete=True):
    folder = args.output / stage
    if not folder.exists():
        if require_complete:
            raise ValueError(f"{stage} stage absent")
        return
    sealed = json.loads((folder / "seal.json").read_text())
    for name, expected in sealed["artifacts"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"changed {stage} artifact: {name}")
    registered = json.loads((folder / "registration.json").read_text())["identities"]
    base = {k: v for k, v in current.items() if not k.startswith(("prepare/", "fit/"))}
    prior = {k: v for k, v in registered.items() if not k.startswith(("prepare/", "fit/"))}
    if base != prior:
        raise ValueError(f"{stage} produced under different frozen source/inputs")
    report = json.loads((folder / "report.json").read_text())
    if require_complete and (
        report.get("status") != "completed" or not report.get("integrity_verified_after")
    ):
        raise ValueError(f"{stage} stage not complete and verified")


def optional_fit_status(args, expected):
    try:
        verify_producer(args, "fit", expected)
        return dict(eligible=True)
    except (ValueError, OSError, KeyError, TypeError) as error:
        return dict(eligible=False, reason=f"{type(error).__name__}: {error}")


def verify_payloads(dataset):
    manifest = json.loads((dataset / "meta/jepa_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if digest(dataset / name) != expected:
            raise ValueError(f"dataset encoded payload changed: {name}")


def worker(args):
    import resource

    import numpy as np

    np.seterr(invalid="raise", divide="raise", over="raise")
    sys.path.insert(0, str(ROOT / "src"))
    folder = args.output / args.stage
    report = dict(
        stage=args.stage,
        status="running",
        test_images_decoded=False,
        planned_candidates=["mean", "nn", "neural"],
    )
    if args.stage == "evaluate" and (args.output / "prepare/roots.json").exists():
        report["roots"] = [
            dict(root_id=r["root_id"], status="not_started")
            for r in json.loads((args.output / "prepare/roots.json").read_text())
        ]
    start = (args.wall, args.mono)

    def check():
        if elapsed(start) >= BUDGETS[args.stage] - 5:
            raise TimeoutError("stage wall budget exhausted; finalization reserved")

    expected = json.loads((folder / "registration.json").read_text())["identities"]
    try:
        check()
        if identities(args) != expected:
            raise ValueError("stage inputs changed before execution")
        if args.stage != "prepare":
            verify_producer(args, "prepare", expected)
        if args.stage == "evaluate":
            report["fit_validation"] = optional_fit_status(args, expected)
        {"prepare": prepare, "fit": fit, "evaluate": evaluate}[args.stage](args, check, report)
        check()
        if identities(args) != expected:
            raise ValueError("stage inputs changed during execution")
        report.update(status="completed", integrity_verified_after=True)
    except Exception as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        report.update(
            worker_cpu_seconds=time.process_time(),
            wall_seconds=elapsed(start),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1 if sys.platform == "darwin" else 1024),
        )
        write(folder / "report.json", json_arrays(report))
    return 0 if report["status"] == "completed" else 2


def supervise(child, *, budget, clock=elapsed, sleep=time.sleep, kill=os.killpg):
    while True:
        expired = clock() >= budget
        code = child.poll()
        if expired:
            if code is None:
                try:
                    kill(child.pid, 9)
                except ProcessLookupError:
                    pass
            return child.wait(), True
        if code is not None:
            return code, False
        sleep(0.05)


def finalize(args, code, timeout, expected):
    folder = args.output / args.stage
    path = folder / "report.json"
    try:
        report = json.loads(path.read_text()) if path.exists() else dict(status="incomplete")
        if not isinstance(report, dict):
            raise ValueError("worker report is not an object")
    except (ValueError, OSError) as error:
        report = dict(status="incomplete", report_error=str(error))
    report.setdefault("planned_candidates", ["mean", "nn", "neural"])
    if (
        args.stage == "evaluate"
        and "roots" not in report
        and (args.output / "prepare/roots.json").exists()
    ):
        try:
            report["roots"] = [
                dict(root_id=r["root_id"], status="not_started")
                for r in json.loads((args.output / "prepare/roots.json").read_text())
            ]
        except (ValueError, OSError, KeyError, TypeError) as error:
            report["root_ledger_error"] = str(error)
    valid = (
        code == 0
        and not timeout
        and report.get("status") == "completed"
        and report.get("integrity_verified_after") is True
    )
    try:
        if identities(args) != expected:
            raise ValueError("post-supervision identity mismatch")
        verify_payloads(args.dataset)
    except Exception as error:
        valid = False
        report["integrity_error"] = str(error)
    valid = valid and elapsed() < BUDGETS[args.stage]
    report.update(
        status="completed" if valid else "incomplete",
        supervisor_returncode=code,
        supervisor_timeout=timeout,
        supervisor_wall_seconds=elapsed(),
        test_images_decoded=False,
    )
    if args.stage == "evaluate" and not valid:
        report["decisions"] = {
            name: dict(status="unverified", reason="incomplete_evaluation")
            for name in ("nn", "neural")
        }
    write(path, report)
    seal_stage(folder)
    if elapsed() >= BUDGETS[args.stage]:
        valid = False
        report.update(
            status="incomplete",
            finalization_deadline_exceeded=True,
            supervisor_wall_seconds=elapsed(),
        )
        if args.stage == "evaluate":
            report["decisions"] = {
                name: dict(status="unverified", reason="finalization_deadline")
                for name in ("nn", "neural")
            }
        write(path, report)
        seal_stage(folder)
    return 0 if valid else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(BUDGETS), required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/apple-branches-v1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/apple-branches-h16-sensor-v1/sensor.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wall", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mono", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("dataset", "checkpoint", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        return worker(args)
    folder = args.output / args.stage
    if folder.exists():
        raise FileExistsError("stage output exists; retries/overwrite are prohibited")
    folder.mkdir(parents=True)
    expected = {}
    try:
        expected = identities(args)
        verify_payloads(args.dataset)
        registration = dict(
            stage=args.stage,
            budget_seconds=BUDGETS[args.stage],
            reserve_seconds=5,
            identities=expected,
            environment=dict(
                python=sys.version,
                packages={
                    n: importlib.metadata.version(n)
                    for n in ("torch", "numpy", "pyarrow", "Pillow")
                },
                threads=4,
            ),
            config=dict(
                seed=0,
                updates=2000,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                gradient_clip=10,
                logvar_bounds=[-6, 3],
                fields=FIELDS,
            ),
        )
        if args.stage == "prepare":
            registration["sample_ledger"] = sample_table(
                json.loads((args.dataset / "meta/jepa_manifest.json").read_text())
            )
        else:
            verify_producer(args, "prepare", expected)
            if args.stage == "evaluate":
                registration["fit_validation"] = optional_fit_status(args, expected)
        write(folder / "registration.json", registration)
    except Exception as error:
        write(
            folder / "report.json",
            dict(status="incomplete", error=f"{type(error).__name__}: {error}"),
        )
        return finalize(args, 2, False, expected)
    if elapsed() >= BUDGETS[args.stage] - 5:
        return finalize(args, 2, True, expected)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--stage",
        args.stage,
        "--dataset",
        str(args.dataset),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--wall",
        str(ENTRY[0]),
        "--mono",
        str(ENTRY[1]),
    ]
    with (folder / "worker.log").open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ
            | {
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "VECLIB_MAXIMUM_THREADS": "4",
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code, timeout = supervise(child, budget=BUDGETS[args.stage] - 5)
    return finalize(args, code, timeout, expected)


if __name__ == "__main__":
    raise SystemExit(main())
