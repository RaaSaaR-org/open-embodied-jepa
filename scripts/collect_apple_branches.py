"""Collect preregistered matched sensor/action branches; no learned-control claim."""

from __future__ import annotations

# ruff: noqa: E402
import time

ENTRY_WALL, ENTRY_MONO = time.time(), time.monotonic()

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_jepa.data import DatasetStore, Episode  # noqa: E402
from embodied_jepa.training import source_identity  # noqa: E402

PARENT_SHA = "d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df"
PHASES = ("orient", "descend", "close", "lift", "transfer", "release_high")
BRANCHES = (
    "recorded",
    "hold",
    "half_motion",
    "reverse_translation",
    "reverse_rotation",
    "up",
    "open",
    "closed",
)
HORIZON = 16


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def elapsed(wall=ENTRY_WALL, mono=ENTRY_MONO):
    return max(time.time() - wall, time.monotonic() - mono)


def make_plan(manifest):
    rows = {r["episode_id"]: r for r in manifest["episodes"]}
    parents = [
        (x, "train")
        for x in sorted(manifest["splits"]["train"])
        if rows[x]["metadata"]["variant"] == "nominal"
    ][:8]
    parents += [(x, "val") for x in sorted(manifest["splits"]["val"])]
    if len(parents) != 11:
        raise ValueError("frozen design requires eight TRAIN and three VAL parents")
    roots, attempts = [], []
    for parent, split in parents:
        labels = rows[parent]["metadata"]["phase_labels"]
        for phase in PHASES:
            starts = [
                i
                for i in range(len(labels) - HORIZON + 1)
                if all(p == phase for p in labels[i : i + HORIZON])
            ]
            start = starts[len(starts) // 2] if starts else None
            root = dict(
                root_id=f"{parent}-{phase}",
                parent_episode_id=parent,
                session_id=rows[parent]["session_id"],
                split=split,
                phase=phase,
                frame_index=start,
            )
            roots.append(root)
            for branch in BRANCHES:
                attempts.append(
                    root
                    | dict(
                        branch=branch,
                        episode_id=f"{root['root_id']}-{branch}",
                        status="not_started",
                    )
                )
    return dict(
        roots=roots,
        attempts=attempts,
        horizon=HORIZON,
        max_wall_seconds=600,
        execution_cutoff_seconds=540,
        parent_dataset_sha256=PARENT_SHA,
        test_decoded=False,
    )


def branch_action(recorded, kind, root_grasp):
    action = np.array(recorded, dtype=np.float32, copy=True)
    if kind == "hold":
        action[:12] = 0
        action[12:] = root_grasp
    elif kind == "half_motion":
        action[6:12] *= 0.5
    elif kind == "reverse_translation":
        action[6:9] *= -1
    elif kind == "reverse_rotation":
        action[9:12] *= -1
    elif kind == "up":
        action[6:12] = 0
        action[8] = 0.25
    elif kind in ("open", "closed"):
        action[13] = -1 if kind == "open" else 1
    elif kind != "recorded":
        raise ValueError("unknown branch")
    return np.clip(action, -1, 1).astype(np.float32)


def identity(source):
    return dict(
        source=source_identity(),
        parent_sha256=source.manifest_hash,
        script_sha256=digest(__file__),
        protocol_sha256=digest(ROOT / "docs/experiments/apple_branches_v1.md"),
        action_sha256=digest(ROOT / "configs/g1_sim_action.json"),
        assets_sha256=digest(ROOT / "assets/manifest.json"),
    )


def capture(robot):
    obs = robot.observe()
    return (
        obs.images["onboard_rgb"][0].copy(),
        obs.robot_state[0].copy(),
        obs.state_mask[0].copy(),
        float(obs.timestamps[0]),
    )


def same_capture(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def snapshot(robot, scorer):
    data = robot.mj.MjData(robot.model)
    robot.mj.mj_copyData(data, robot.model, robot.sim.data)
    return dict(
        data=data,
        targets=robot.sim.targets.copy(),
        grasp=robot._grasp.copy(),
        stopped_reason=robot.sim.stopped_reason,
        scorer={k: copy.deepcopy(v) for k, v in vars(scorer).items() if k != "robot"},
    )


def restore(robot, scorer, saved):
    robot.mj.mj_copyData(robot.sim.data, robot.model, saved["data"])
    robot.sim.targets[:] = saved["targets"]
    robot._grasp[:] = saved["grasp"]
    robot.sim.stopped_reason = saved["stopped_reason"]
    robot._observation = None
    for key, value in saved["scorer"].items():
        setattr(scorer, key, copy.deepcopy(value))
    return capture(robot)


def paired_metrics(root_id, split, phase, endpoints):
    pairs = []
    names = sorted(endpoints)
    for horizon in (8, 16):
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if horizon not in endpoints[left] or horizon not in endpoints[right]:
                    continue
                (a, aa), (b, ba) = endpoints[left][horizon], endpoints[right][horizon]
                rgb_rms = float(np.sqrt(np.mean(((a[0].astype(float) - b[0]) / 255) ** 2)))
                pairs.append(
                    dict(
                        left=left,
                        right=right,
                        horizon=horizon,
                        rgb_rms=rgb_rms,
                        measured_state_rms=float(np.sqrt(np.mean((a[1] - b[1]) ** 2))),
                        applied_action_rms=float(np.sqrt(np.mean((aa - ba) ** 2))),
                        visually_distinct=rgb_rms >= 1 / 255,
                    )
                )
    return dict(root_id=root_id, split=split, phase=phase, pairs=pairs)


def journal_frame(directory, index, frame, **arrays):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"frame-{index:03d}.npz"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, rgb=frame[0], state=frame[1], mask=frame[2], timestamp=frame[3], **arrays)
    temporary.replace(destination)


def worker(source_path, output, plan_path, wall, mono):
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation
    from embodied_jepa.task import AppleToPlateTask

    source = DatasetStore(source_path)
    plan = json.loads(plan_path.read_text())
    frozen = plan["identity"]
    if identity(source) != frozen:
        raise ValueError("frozen source identity changed before worker")
    records = copy.deepcopy(plan["attempts"])
    report_path = output / "branch_report.json"
    report = dict(
        status="running",
        training_ready=False,
        attempts=records,
        identity=frozen,
        test_decoded=False,
        matched_root_metrics=[],
    )
    cpu = time.process_time()

    def persist():
        report["true_wall_seconds"] = elapsed(wall, mono)
        report["process_cpu_seconds"] = time.process_time() - cpu
        write(report_path, report)

    # Public encoded-copy API preserves TEST payloads without image decoding.
    store = source.fork_unsealed(
        output,
        provenance={
            "producer": "collect_apple_branches.py",
            "source": frozen,
            "task_specific_development": True,
            "plan": str(plan_path),
        },
    )
    import shutil

    waypoint_index = source.root / "train_waypoints.json"
    if waypoint_index.exists():
        shutil.copyfile(waypoint_index, output / "parent_train_waypoints.json")
        shutil.copytree(source.root / "train_waypoints", output / "train_waypoints")
    assignments = copy.deepcopy(source.manifest["splits"])
    persist()
    for parent_id in dict.fromkeys(root["parent_episode_id"] for root in plan["roots"]):
        parent_roots = [r for r in plan["roots"] if r["parent_episode_id"] == parent_id]
        parent_records = [r for r in records if r["parent_episode_id"] == parent_id]
        if elapsed(wall, mono) >= 540:
            break
        episode = source.read_episode(parent_id)
        requests = np.asarray(episode.metadata["requested_actions"], np.float32)
        trial = episode.metadata["trial"]
        robot = G1Embodiment(
            MuJoCoSimulation(object_kind="apple", container_kind="plate", width=96, height=96)
        )
        try:
            robot.reset(trial["seed"], object_xy=trial["object_xy"], plate_xy=trial["plate_xy"])
            scorer = AppleToPlateTask(robot)
            observed = capture(robot)
            step = 0
            for root in parent_roots:
                siblings = [r for r in parent_records if r["root_id"] == root["root_id"]]
                start = root["frame_index"]
                if start is None:
                    for row in siblings:
                        row["status"] = "unavailable_phase"
                    persist()
                    continue
                if elapsed(wall, mono) >= 540:
                    break
                while step < start:
                    if elapsed(wall, mono) >= 540:
                        raise TimeoutError("replay budget exhausted")
                    result = robot.execute(requests[step])
                    if result.applied_action is None:
                        raise ValueError(f"replay rejected step {step}: {result.reason}")
                    if not np.allclose(
                        result.applied_action, episode.actions[step], atol=2e-6, rtol=0
                    ):
                        raise ValueError(f"replay applied action mismatch at {step}")
                    scorer.evaluate()
                    observed = capture(robot)
                    step += 1
                    if not np.allclose(observed[1], episode.robot_states[step], atol=1e-5, rtol=0):
                        raise ValueError(f"replay sensor mismatch at {step}")
                if (
                    not np.array_equal(observed[0], episode.observations["onboard_rgb"][start])
                    or not np.array_equal(observed[2], episode.state_mask[start])
                    or not np.isclose(observed[3], episode.timestamps[start], atol=1e-9, rtol=0)
                ):
                    raise ValueError("replayed root image/mask/time differs from source")
                endpoints = {}
                saved = snapshot(robot, scorer)
                reference = observed
                # Repeated continuation verifies hidden-state restoration beyond initial pixels.
                probes = []
                for _ in range(2):
                    if elapsed(wall, mono) >= 540:
                        raise TimeoutError("restoration probe budget exhausted")
                    if not same_capture(restore(robot, scorer, saved), reference):
                        raise ValueError("restored root sensors differ")
                    result = robot.execute(requests[start])
                    probes.append((result.applied_action, capture(robot)))
                if (
                    probes[0][0] is None
                    or probes[1][0] is None
                    or not np.array_equal(probes[0][0], probes[1][0])
                    or not same_capture(probes[0][1], probes[1][1])
                ):
                    raise ValueError("restored continuation is not deterministic")
                for record in siblings:
                    if elapsed(wall, mono) >= 540:
                        break
                    record["status"] = "running"
                    persist()
                    frames = [restore(robot, scorer, saved)]
                    if not same_capture(frames[0], reference):
                        raise ValueError("sibling initial sensors differ")
                    actions, desired, raw, scores = [], [], [], []
                    journal = output / "branch_journal" / record["episode_id"]
                    journal_frame(journal, 0, frames[0])
                    record["journal"] = str(journal.relative_to(output))
                    persist()
                    for offset in range(HORIZON):
                        if elapsed(wall, mono) >= 540:
                            record["status"] = "budget_truncated"
                            break
                        command = branch_action(
                            requests[start + offset], record["branch"], saved["grasp"]
                        )
                        write(
                            journal / "command.json",
                            {"index": offset + 1, "status": "pending", "request": command.tolist()},
                        )
                        try:
                            result = robot.execute(command)
                            if result.applied_action is None:
                                record.update(
                                    status="command_rejected",
                                    failure=result.reason,
                                    rejected_request=command.tolist(),
                                )
                                write(
                                    journal / "command.json",
                                    {"index": offset + 1, "status": "rejected"},
                                )
                                break
                            measured = capture(robot)
                            physical = robot.denormalize_action(result.applied_action)
                            journal_frame(
                                journal,
                                offset + 1,
                                measured,
                                action=result.applied_action,
                                request=command,
                                raw_action=physical,
                            )
                            actions.append(result.applied_action.copy())
                            desired.append(command)
                            raw.append(physical)
                            frames.append(measured)
                            scores.append(scorer.evaluate())
                        except Exception as exc:
                            record.update(
                                status="runtime_error",
                                failure=f"{type(exc).__name__}: {exc}",
                                possible_unobserved_execution=True,
                            )
                            # Captured transitions remain durable and canonicalizable.
                            if len(scores) < len(actions):
                                scores.append({"unavailable": True, "reason": str(exc)})
                            break
                    if record["status"] == "running":
                        record["status"] = "completed"
                    record["executed_steps"] = len(actions)
                    endpoints[record["branch"]] = {
                        h: (frames[h], np.asarray(actions[:h]))
                        for h in (8, 16)
                        if len(actions) >= h
                    }
                    record["final_score"] = scores[-1] if scores else None
                    record["root_sensor_sha256"] = hashlib.sha256(
                        reference[0].tobytes() + reference[1].tobytes()
                    ).hexdigest()
                    if actions:
                        images, states, masks, timestamps = zip(*frames, strict=True)
                        branch = Episode(
                            episode_id=record["episode_id"],
                            session_id=episode.session_id,
                            task=episode.task,
                            object_id=episode.object_id,
                            container_id=episode.container_id,
                            observations={"onboard_rgb": np.asarray(images)},
                            robot_states=np.asarray(states),
                            state_mask=np.asarray(masks),
                            timestamps=np.asarray(timestamps),
                            actions=np.asarray(actions),
                            raw_actions=np.asarray(raw),
                            state_schema=episode.state_schema,
                            terminated=False,
                            truncated=True,
                            metadata=dict(
                                parent_episode_id=parent_id,
                                parent_manifest_sha256=PARENT_SHA,
                                root_id=root["root_id"],
                                root_frame=start,
                                branch=record["branch"],
                                phase_labels=[root["phase"]] * len(actions),
                                requested_actions=np.asarray(desired).tolist(),
                                stage_scores=scores,
                                final_score=record["final_score"],
                                termination=record["status"],
                                failure=record.get("failure"),
                                root_sensor_sha256=record["root_sensor_sha256"],
                                collection_success=False,
                                oracle_collection_only=False,
                                recorded_demonstration_continuation=True,
                            ),
                        )
                        try:
                            store.write_episode(branch)
                            assignments[root["split"]].append(branch.episode_id)
                        except Exception as exc:
                            record.update(
                                status="storage_error", failure=f"{type(exc).__name__}: {exc}"
                            )
                    persist()
                report.setdefault("matched_root_metrics", []).append(
                    paired_metrics(root["root_id"], root["split"], root["phase"], endpoints)
                )
                observed = restore(robot, scorer, saved)
        except Exception as exc:
            for row in parent_records:
                if row["status"] in ("not_started", "running"):
                    row.update(
                        status="replay_or_runtime_error", failure=f"{type(exc).__name__}: {exc}"
                    )
            persist()
        finally:
            robot.close()
    for record in records:
        if record["status"] == "not_started":
            record["status"] = "not_started_budget"
    source.verify()
    if identity(DatasetStore(source_path)) != frozen:
        raise ValueError("source changed during collection")
    store.freeze_split_assignments(
        assignments,
        provenance={"parent_sha256": PARENT_SHA, "grouping": "original parent session"},
        heldout_combinations=(),
    )
    store.fit_normalization()
    store.verify()
    source.verify()
    if identity(DatasetStore(source_path)) != frozen:
        raise ValueError("source changed during corpus sealing")
    incomplete = any(
        r["status"] not in ("completed", "command_rejected", "unavailable_phase") for r in records
    )
    contrast = {}
    for split in ("train", "val"):
        available = [
            r for r in plan["roots"] if r["split"] == split and r["frame_index"] is not None
        ]
        passed = 0
        for root in available:
            complete = sum(
                r["root_id"] == root["root_id"] and r.get("executed_steps", 0) >= 16
                for r in records
            )
            metric = next(
                (
                    m
                    for m in report.get("matched_root_metrics", [])
                    if m["root_id"] == root["root_id"]
                ),
                {"pairs": []},
            )
            distinct = sum(p["horizon"] == 16 and p["visually_distinct"] for p in metric["pairs"])
            passed += complete >= 4 and distinct >= 2
        contrast[split] = dict(
            available_roots=len(available),
            passing_roots=passed,
            passed=bool(available) and passed / len(available) >= 0.75,
        )
    report["action_contrast_gate"] = contrast
    report["action_contrast_gate_passed"] = all(c["passed"] for c in contrast.values())
    report.update(
        status="incomplete" if incomplete else "completed",
        training_ready=not incomplete and report["action_contrast_gate_passed"],
        dataset_sha256=store.manifest_hash,
        split_counts={k: len(v) for k, v in assignments.items()},
        integrity_verified_after=True,
    )
    if not incomplete and not report["training_ready"]:
        report["status"] = "completed_not_training_ready"
    persist()
    return 0 if report["training_ready"] else 2


def finalize(output, plan, returncode, timed_out):
    report_path = output / "branch_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    existing = {r["episode_id"]: r for r in report.get("attempts", [])}
    records = [existing.get(r["episode_id"], copy.deepcopy(r)) for r in plan["attempts"]]
    for row in records:
        if row["status"] == "running":
            journal = output / "branch_journal" / row["episode_id"]
            count = max(0, len(list(journal.glob("frame-*.npz"))) - 1)
            row.update(
                status="interrupted_unknown_execution",
                executed_steps=None,
                durable_captured_transitions=count,
                journal=str(journal.relative_to(output)),
            )
        elif row["status"] == "not_started":
            row["status"] = (
                "not_started_supervisor_timeout" if timed_out else "not_started_worker_failure"
            )
    ready = returncode == 0 and not timed_out and report.get("training_ready", False)
    status = "completed" if ready else "incomplete"
    if not timed_out and report.get("status") == "completed_not_training_ready" and returncode == 2:
        status = "completed_not_training_ready"
    report.update(
        attempts=records,
        status=status,
        training_ready=ready,
        returncode=returncode,
        supervisor_timeout=timed_out,
        supervisor_wall_seconds=elapsed(),
    )
    write(report_path, report)
    return 0 if ready else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/apple-task-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wall", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mono", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    source_path, output = args.source.resolve(), args.output.resolve()
    plan_path = output.with_name(output.name + "-plan.json")
    if args.worker:
        return worker(source_path, output, plan_path, args.wall, args.mono)
    if output.exists() or plan_path.exists():
        raise FileExistsError("refusing to overwrite branch collection or plan")
    source = DatasetStore(source_path)
    if source.manifest_hash != PARENT_SHA:
        raise ValueError("parent manifest differs from protocol")
    plan = make_plan(source.manifest) | {"identity": identity(source)}
    write(plan_path, plan)
    if elapsed() >= 595:
        return finalize(output, plan, 2, True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source",
        str(source_path),
        "--output",
        str(output),
        "--wall",
        str(ENTRY_WALL),
        "--mono",
        str(ENTRY_MONO),
    ]
    with plan_path.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        )
        timed_out = False
        while process.poll() is None:
            if elapsed() >= 595:
                process.kill()
                timed_out = True
                break
            time.sleep(0.1)
        returncode = process.wait()
    return finalize(output, plan, returncode, timed_out)


if __name__ == "__main__":
    raise SystemExit(main())
