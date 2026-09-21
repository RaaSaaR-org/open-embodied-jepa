"""Prepare a frozen hybrid-cost bundle and audit saved forecasts under fixed budgets."""

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
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {"prepare": 120.0, "audit": 60.0}
PROTOCOL = "docs/experiments/apple_aligned_control_v1.md"
STRIDE = 28
RECIPE = "train_parent_median_stride28_v1"
PARITY_ATOL = 1e-7
PARITY_RTOL = 1e-5


def elapsed():
    return max(0.0, time.time() - ENTRY[0], time.monotonic() - ENTRY[1])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def helper():
    spec = importlib.util.spec_from_file_location(
        "task041_helpers", ROOT / "scripts/apple_goal_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_ledger(manifest, *, expected_parents=26):
    """Metadata-only selection, before any decoding; keep settled/zero-change pairs."""
    rows = {r["episode_id"]: r for r in manifest["episodes"]}
    train = manifest["splits"]["train"]
    if len(train) != len(set(train)):
        raise ValueError("duplicate TRAIN membership")
    if set(train) & (set(manifest["splits"]["val"]) | set(manifest["splits"]["test"])):
        raise ValueError("TRAIN split overlaps held-out episodes")
    originals = [rows[n] for n in sorted(train) if "parent_episode_id" not in rows[n]["metadata"]]
    if (
        len(originals) != expected_parents
        or len({r["session_id"] for r in originals}) != expected_parents
    ):
        raise ValueError("expected distinct original TRAIN parent sessions")
    result = []
    for row in originals:
        pairs = [[t, t + STRIDE] for t in range(0, row["length"] - STRIDE, STRIDE)]
        if not pairs:
            raise ValueError("every parent needs a complete stride28 pair")
        result.append(dict(episode_id=row["episode_id"], session_id=row["session_id"], pairs=pairs))
    return result


def reduce_scales(parent_distances):
    import numpy as np

    medians = []
    for parent in parent_distances:
        values = np.asarray(parent["distances"], dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or not len(values)
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError("each parent requires finite nonnegative visual/pose distances")
        medians.append(np.median(values, axis=0))
    if not medians:
        raise ValueError("no parent distances")
    scales = np.median(np.stack(medians), axis=0)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("degenerate calibration; no floors or replacement statistic allowed")
    return dict(visual=float(scales[0]), pose=float(scales[1])), [v.tolist() for v in medians]


def seal(folder):
    write(
        folder / "seal.json",
        {
            "artifacts": {
                str(p.relative_to(folder)): digest(p)
                for p in sorted(folder.rglob("*"))
                if p.is_file() and p.name not in ("seal.json", "worker.log")
            }
        },
    )


def verify_seal(folder, *, require_complete=True):
    value = read(folder / "seal.json")
    for name, expected in value["artifacts"].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or digest(path) != expected:
            raise ValueError(f"changed producer artifact: {name}")
    report = read(folder / "report.json")
    if require_complete and (
        report.get("status") != "completed" or not report.get("integrity_verified_after")
    ):
        raise ValueError("producer did not complete with verified inputs")
    return report


def identities(args):
    files = {
        str(p.relative_to(ROOT)): p for p in sorted((ROOT / "src/embodied_jepa").rglob("*.py"))
    }
    files.update(
        script=Path(__file__),
        protocol=ROOT / PROTOCOL,
        ranking_helper=ROOT / "scripts/apple_goal_alignment.py",
        checkpoint_loader=ROOT / "scripts/evaluate_apple.py",
        dataset=args.dataset / "meta/jepa_manifest.json",
        checkpoint=args.checkpoint,
    )
    for p in sorted(args.alignment.rglob("*")):
        if p.is_file() and p.name != "worker.log":
            files["alignment/" + str(p.relative_to(args.alignment))] = p
    if args.stage == "audit":
        for p in sorted((args.output / "prepare").rglob("*")):
            if p.is_file() and p.name != "worker.log":
                files["prepare/" + str(p.relative_to(args.output / "prepare"))] = p
    result = {k: digest(p) for k, p in files.items()}
    old = helper()
    if result["dataset"] != old.DATA_SHA or result["checkpoint"] != old.CHECKPOINT_SHA:
        raise ValueError("registered corpus or world model differs")
    result["revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return result


def verify_inputs(args):
    old = helper()
    old.verify_payloads(args.dataset)  # Encoded bytes only, including TEST; no decoding.
    reports = {s: verify_seal(args.alignment / s) for s in ("prepare", "fit", "evaluate")}
    if (
        reports["fit"].get("updates") != 2000
        or reports["evaluate"].get("decisions", {}).get("neural", {}).get("status") != "passed"
    ):
        raise ValueError("frozen neural head lacks completed TASK041 eligibility")
    historical_base = None
    for stage in reports:
        prior = read(args.alignment / stage / "registration.json")["identities"]
        base = {k: v for k, v in prior.items() if not k.startswith(("prepare/", "fit/"))}
        if historical_base is None:
            historical_base = base
        elif base != historical_base:
            raise ValueError("TASK041 stages have different historical source/input identities")
        if prior["dataset"] != old.DATA_SHA or prior["checkpoint"] != old.CHECKPOINT_SHA:
            raise ValueError("TASK041 head does not belong to this corpus/world model")
        for name, expected in prior.items():
            if name.startswith(("prepare/", "fit/")) and digest(args.alignment / name) != expected:
                raise ValueError("TASK041 producer artifact chain differs")
    # Historical config may legitimately differ after adding a backend. Compare
    # producer stages to each other; immutable child compatibility is its loader's job.
    if historical_base["script"] != digest(ROOT / "scripts/apple_goal_alignment.py"):
        raise ValueError("saved cohort metric helper is no longer the frozen TASK041 script")
    if args.stage == "audit":
        verify_seal(args.output / "prepare")
        previous = read(args.output / "prepare/registration.json")["identities"]
        current = identities(args)
        if {k: v for k, v in current.items() if not k.startswith("prepare/")} != previous:
            raise ValueError("bundle produced under different source or inputs")


def prepare(args, check, report):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    old = helper()
    folder = args.output / "prepare"
    ledger = pair_ledger(read(args.dataset / "meta/jepa_manifest.json"))
    write(folder / "pairs.json", ledger)
    report.update(
        planned_parents=[r["episode_id"] for r in ledger],
        decoded_episode_ids=[],
        pair_ledger_sha256_before_decode=digest(folder / "pairs.json"),
    )
    write(folder / "report.json", report)
    store, model = old.load_model(args)
    indices = np.asarray(old.resolve_fields(store.state_schema))
    mean = model.sensor_mean[model.visual_dimension + indices].numpy()
    scale = model.sensor_scale[model.visual_dimension + indices].numpy()
    distances = []
    for parent_index, row in enumerate(ledger):
        check()
        report["decode_in_progress"] = row["episode_id"]
        write(folder / "report.json", report)
        ep = store.read_episode(row["episode_id"])
        frames = sorted({i for pair in row["pairs"] for i in pair})
        if not ep.state_mask[frames][:, indices].all():
            raise ValueError("missing observed calibration joint")
        x = (
            model.encode_goal(
                {model.config["camera"]: ep.observations[model.config["camera"]][frames]}
            )
            .values.numpy()
            .astype(np.float64)
        )
        q = ((ep.robot_states[frames][:, indices] - mean) / scale).astype(np.float64)
        # Preserve the exact operands used below, including zeros and settled
        # frames, so evidence checks need no decoding or model inference.
        old.save_npz(
            folder / f"signals-{parent_index:02d}.npz",
            episode_id=np.asarray(row["episode_id"]),
            frame_indices=np.asarray(frames, dtype=np.int64),
            visual_features=x,
            normalized_positions=q,
            field_indices=indices,
            position_mean=mean,
            position_scale=scale,
        )
        index = {f: i for i, f in enumerate(frames)}
        values = [
            [
                float(np.square(x[index[a]] - x[index[b]]).mean()),
                float(np.square(q[index[a]] - q[index[b]]).mean()),
            ]
            for a, b in row["pairs"]
        ]
        distances.append(dict(episode_id=row["episode_id"], distances=values))
        write(folder / "distances.json", distances)
        report["decoded_episode_ids"].append(row["episode_id"])
        report["decode_in_progress"] = None
        write(folder / "report.json", report)
    scales, medians = reduce_scales(distances)
    complete_pairs = {
        "pairs": [
            dict(
                parent_id=row["episode_id"],
                frame0=a,
                frame1=b,
                visual_mse=values[0],
                pose_mse=values[1],
            )
            for row, distance in zip(ledger, distances, strict=True)
            for (a, b), values in zip(row["pairs"], distance["distances"], strict=True)
        ]
    }
    write(folder / "calibration_ledger.json", complete_pairs)
    calibration = dict(
        format_version=1,
        recipe=RECIPE,
        stride=STRIDE,
        weights={"visual": 0.5, "pose": 0.5},
        scales=scales,
        parent_ids=[r["episode_id"] for r in ledger],
        parent_medians=medians,
        fields=list(old.FIELDS),
        field_indices=indices.tolist(),
        sensor_checkpoint_sha256=digest(args.checkpoint),
        goal_head_sha256=digest(args.alignment / "fit/head.pt"),
        dataset_hash=store.manifest_hash,
        ledger_sha256=digest(folder / "calibration_ledger.json"),
        packaging=dict(
            source_sha256=digest(Path(__file__)),
            protocol_sha256=digest(ROOT / PROTOCOL),
            source_revision=read(folder / "registration.json")["identities"]["revision"],
        ),
    )
    write(folder / "calibration.json", calibration)
    check()
    if digest(folder / "pairs.json") != report["pair_ledger_sha256_before_decode"]:
        raise ValueError("calibration pair ledger changed while decoding")
    package_bundle(args, folder, calibration)
    report.update(
        scales=scales,
        parent_count=len(ledger),
        pair_count=sum(len(r["pairs"]) for r in ledger),
        bundle="bundle/aligned.pt",
    )


def package_bundle(args, folder, calibration):
    from embodied_jepa.models.aligned_sensor import package_bundle as export

    return export(
        folder / "bundle/aligned.pt",
        sensor_checkpoint=args.checkpoint,
        head_checkpoint=args.alignment / "fit/head.pt",
        calibration_json=folder / "calibration.json",
        provenance_files={
            "dataset_manifest": args.dataset / "meta/jepa_manifest.json",
            "goal_registration": args.alignment / "fit/registration.json",
            "goal_fit_report": args.alignment / "fit/report.json",
            "goal_fit_seal": args.alignment / "fit/seal.json",
            "goal_train_samples": args.alignment / "prepare/train_samples.json",
            "goal_source": ROOT / "scripts/apple_goal_alignment.py",
            "calibration_ledger": folder / "calibration_ledger.json",
        },
    )


def hybrid_gate(ranking):
    import numpy as np

    pixel = ranking["parents"].get("pixel_pred", {})
    hybrid = ranking["parents"].get("hybrid_pred", {})
    if not ranking["sufficient_coverage"] or len(pixel) != 3 or set(hybrid) != set(pixel):
        return dict(status="unverified", reason="insufficient_coverage")
    parents = sorted(pixel)
    deltas = {p: hybrid[p] - pixel[p] for p in parents}
    delta = float(np.mean(list(deltas.values())))
    draw = np.random.default_rng(0).integers(0, 3, (2000, 3))

    def interval(values):
        return np.quantile(np.asarray(values)[draw].mean(1), [0.025, 0.975]).tolist()

    return dict(
        status="passed" if delta >= 0.05 and all(v > 0 for v in deltas.values()) else "failed",
        ranking_delta=delta,
        parent_deltas=deltas,
        ranking_ci95=interval([hybrid[p] for p in parents]),
        delta_ci95=interval(list(deltas.values())),
    )


def audit(args, check, report):
    import numpy as np

    from embodied_jepa.models.aligned_sensor import hybrid_cost

    old = helper()
    folder = args.output / "audit"
    prep = args.alignment / "prepare"
    calibration = read(args.output / "prepare/calibration.json")
    scales = calibration["scales"]
    rows = read(prep / "val_samples.json")
    roots = read(prep / "roots.json")
    original_cases = read(args.alignment / "evaluate/primary_cases.json")
    if len(rows) != 360 or len(roots) != 18 or any(r["split"] != "val" for r in rows):
        raise ValueError("saved VAL cohort is incomplete")
    with np.load(prep / "val_features.npz", allow_pickle=False) as archive:
        cache = {k: archive[k].copy() for k in archive.files}
    features, labels = cache["features"], cache["targets"]
    goals = np.asarray(
        read(args.alignment / "evaluate/goal_estimates.json")["means"]["neural"], dtype=np.float32
    )
    if goals.shape != labels.shape or not np.isfinite(goals).all():
        raise ValueError("saved neural goal means incompatible")
    q_indices = cache["field_indices"] + 1728
    parents = sorted({r["parent"] for r in rows})
    case_maps = {
        h: {(r["root_id"], r["goal"]["episode_id"], r["goal"]["frame"]): r for r in cases}
        for h, cases in original_cases.items()
    }
    cases = {"8": [], "16": []}
    parity_max = 0.0
    for index, root in enumerate(roots):
        check()
        with np.load(
            args.alignment / f"evaluate/prediction-{index:02d}.npz", allow_pickle=False
        ) as archive:
            prediction = archive["sensor_prediction"].copy()
            if not np.array_equal(q_indices, archive["q_indices"]):
                raise ValueError("saved forecast field mapping differs")
        if prediction.shape != (8, 16, 1814) or not np.isfinite(prediction).all():
            raise ValueError("invalid saved sensor forecast")
        costs_for_root = []
        for horizon in (8, 16):
            h = str(horizon)
            endpoints = cache["root_endpoints"][index, :, 0 if horizon == 8 else 1]
            measured = labels[endpoints]
            goal_ids = [
                i
                for i, r in enumerate(rows)
                if r["source"] == "original"
                and r["phase"] == root["phase"]
                and r["parent"] != root["parent"]
            ]
            if len(goal_ids) != 8:
                raise ValueError("cross-parent goal roster changed")
            for goal in goal_ids:
                costs = {}
                components = {}
                for tag, visual, pose in [
                    (
                        "pred",
                        prediction[:, horizon - 1, :1728],
                        prediction[:, horizon - 1, q_indices],
                    ),
                    ("measured", features[endpoints], measured),
                ]:
                    dv = np.square(visual - features[goal]).mean(1)
                    dq = np.square(pose - goals[goal]).mean(1)
                    direct = np.float32(0.5) * dv / np.float32(scales["visual"]) + np.float32(
                        0.5
                    ) * dq / np.float32(scales["pose"])
                    runtime = hybrid_cost(dv, dq, scales)
                    if (
                        not np.isfinite(runtime).all()
                        or np.asarray(runtime).shape != (8,)
                        or not np.allclose(runtime, direct, rtol=PARITY_RTOL, atol=PARITY_ATOL)
                    ):
                        raise ValueError("runtime hybrid formula parity failed")
                    parity_max = max(parity_max, float(np.max(np.abs(runtime - direct))))
                    costs.update(
                        {"pixel_" + tag: dv, "neural_" + tag: dq, "hybrid_" + tag: runtime}
                    )
                    components[tag] = dict(
                        visual=dv.tolist(),
                        pose=dq.tolist(),
                        hybrid=np.asarray(runtime).tolist(),
                        direct=direct.tolist(),
                    )
                pairs = old.ranking_pairs(measured, labels[goal], costs)
                key = (root["root_id"], rows[goal]["episode_id"], rows[goal]["frame"])
                historical = case_maps[h][key]
                for actual, previous in zip(pairs, historical["pairs"], strict=True):
                    for field in (
                        "i",
                        "j",
                        "eligible",
                        "exclusion",
                        "label_i",
                        "label_j",
                        "arm_rms",
                    ):
                        if actual[field] != previous[field]:
                            raise ValueError("TASK041 pair eligibility or labels changed")
                    if (
                        actual["eligible"]
                        and actual["scores"]["pixel_pred"] != previous["scores"]["pixel_pred"]
                    ):
                        raise ValueError("pixel comparator no longer matches TASK041")
                row = dict(
                    root_id=root["root_id"],
                    parent=root["parent"],
                    phase=root["phase"],
                    goal=historical["goal"],
                    costs={k: v.tolist() for k, v in costs.items()},
                    pairs=pairs,
                )
                cases[h].append(row)
                costs_for_root.append(
                    dict(horizon=horizon, goal=historical["goal"], components=components)
                )
        write(folder / f"costs-{index:02d}.json", costs_for_root)
        # Keep computed cases durable before acknowledging a completed root.
        write(folder / "cases.json", cases)
        report["roots"][index]["status"] = "completed"
        write(folder / "report.json", report)
    rankings = {h: old.aggregate_rankings(value, parents) for h, value in cases.items()}
    if any(len(value) != 144 for value in cases.values()):
        raise ValueError("missing planned root-goal case")
    report.update(
        rankings=rankings,
        decision=hybrid_gate(rankings["16"]),
        formula_parity=dict(
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
            max_absolute_error=parity_max,
            comparison="float32 saved-array formula versus runtime hybrid_cost helper",
            runtime_source_sha256=digest(ROOT / "src/embodied_jepa/models/aligned_sensor.py"),
        ),
        development_selection=True,
        inference_executed=False,
    )


def worker(args):
    import resource

    import numpy as np

    np.seterr(over="raise", invalid="raise", divide="raise")
    sys.path.insert(0, str(ROOT / "src"))
    folder = args.output / args.stage
    report = dict(stage=args.stage, status="running", test_decoded=False, physics_executed=False)
    if args.stage == "audit":
        report["roots"] = [
            dict(root_id=r["root_id"], status="not_started")
            for r in read(args.alignment / "prepare/roots.json")
        ]

    def check():
        used = max(time.time() - args.wall, time.monotonic() - args.mono)
        if used >= BUDGETS[args.stage] - 5:
            raise TimeoutError("stage cutoff reserves finalization")

    expected = read(folder / "registration.json")["identities"]
    try:
        check()
        if identities(args) != expected:
            raise ValueError("inputs changed before worker")
        verify_inputs(args)
        {"prepare": prepare, "audit": audit}[args.stage](args, check, report)
        check()
        if identities(args) != expected:
            raise ValueError("inputs changed during worker")
        report.update(status="completed", integrity_verified_after=True)
    except Exception as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        report.update(
            worker_cpu_seconds=time.process_time(),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1 if sys.platform == "darwin" else 1024),
        )
        write(folder / "report.json", report)
    return 0 if report["status"] == "completed" else 2


def planned_ledger(args):
    if args.stage == "prepare":
        ledger = pair_ledger(read(args.dataset / "meta/jepa_manifest.json"))
        return dict(planned_parents=[r["episode_id"] for r in ledger], planned_parent_count=26)
    return dict(
        roots=[
            dict(root_id=r["root_id"], status="not_started")
            for r in read(args.alignment / "prepare/roots.json")
        ],
        planned_root_count=18,
    )


def finalize(args, code, timeout, expected):
    folder = args.output / args.stage
    try:
        report = read(folder / "report.json")
        if not isinstance(report, dict):
            raise ValueError("worker report is not an object")
    except (ValueError, OSError):
        report = dict(status="incomplete")
    try:
        registered = (
            read(folder / "registration.json") if (folder / "registration.json").exists() else {}
        )
        planned = registered.get("planned") or planned_ledger(args)
        for key, value in planned.items():
            report.setdefault(key, value)
    except Exception as error:
        report["planned_ledger_error"] = str(error)
        report.setdefault(
            "planned_parent_count" if args.stage == "prepare" else "planned_root_count",
            26 if args.stage == "prepare" else 18,
        )
    valid = (
        code == 0
        and not timeout
        and report.get("status") == "completed"
        and report.get("integrity_verified_after") is True
    )
    try:
        if identities(args) != expected:
            raise ValueError("post-run input identity drift")
        verify_inputs(args)
    except Exception as error:
        report["integrity_error"] = str(error)
        valid = False
    valid = valid and elapsed() < BUDGETS[args.stage]
    report.update(
        status="completed" if valid else "incomplete",
        supervisor_returncode=code,
        supervisor_timeout=timeout,
        supervisor_wall_seconds=elapsed(),
    )
    if args.stage == "audit" and not valid:
        report["decision"] = dict(status="unverified", reason="incomplete_audit")
    write(folder / "report.json", report)
    seal(folder)
    if elapsed() >= BUDGETS[args.stage]:
        valid = False
        report.update(
            status="incomplete",
            finalization_deadline_exceeded=True,
            supervisor_wall_seconds=elapsed(),
        )
        if args.stage == "audit":
            report["decision"] = dict(status="unverified", reason="finalization_deadline")
        write(folder / "report.json", report)
        seal(folder)
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
    parser.add_argument("--alignment", type=Path, default=ROOT / "outputs/apple-goal-alignment-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wall", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mono", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("dataset", "checkpoint", "alignment", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        return worker(args)
    folder = args.output / args.stage
    if folder.exists():
        raise FileExistsError("output stage exists; no retries or overwrite")
    folder.mkdir(parents=True)
    expected = {}
    try:
        planned = planned_ledger(args)
        write(folder / "report.json", dict(stage=args.stage, status="not_started", **planned))
        expected = identities(args)
        verify_inputs(args)
        write(
            folder / "registration.json",
            dict(
                planned=planned,
                environment=dict(
                    python=sys.version,
                    platform=platform.platform(),
                    device="cpu",
                    threads=4,
                    packages={
                        name: importlib.metadata.version(name)
                        for name in ("numpy", "torch", "pyarrow", "Pillow")
                    },
                ),
                stage=args.stage,
                identities=expected,
                budget_seconds=BUDGETS[args.stage],
                reserve_seconds=5,
                recipe=RECIPE,
                stride=STRIDE,
                weights=[0.5, 0.5],
                parity_atol=PARITY_ATOL,
                parity_rtol=PARITY_RTOL,
            ),
        )
    except Exception as error:
        write(folder / "report.json", dict(status="incomplete", error=str(error)))
        return finalize(args, 2, False, expected)
    if elapsed() >= BUDGETS[args.stage] - 5:
        return finalize(args, 2, True, expected)
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        args.stage,
        "--worker",
        "--dataset",
        str(args.dataset),
        "--checkpoint",
        str(args.checkpoint),
        "--alignment",
        str(args.alignment),
        "--output",
        str(args.output),
        "--wall",
        str(ENTRY[0]),
        "--mono",
        str(ENTRY[1]),
    ]
    with (folder / "worker.log").open("x") as stream:
        child = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=os.environ
            | {
                n: "4"
                for n in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code, timeout = helper().supervise(child, budget=BUDGETS[args.stage] - 5, clock=elapsed)
    return finalize(args, code, timeout, expected)


if __name__ == "__main__":
    raise SystemExit(main())
