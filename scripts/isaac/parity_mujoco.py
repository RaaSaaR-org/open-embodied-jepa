"""MuJoCo side of the IsaacTransport / MuJoCo transport parity check, and the comparison.

Development tooling (TASK-025), not a gated experiment and not a manipulation result. Run on
the host after ``parity_isaac.py``:

    uv run --no-sync python scripts/isaac/parity_mujoco.py \\
        --isaac outputs/<isaac-run>/run --output outputs/<isaac-run>/parity

It replays exactly the joint targets recorded in ``isaac_trace.npz`` through
``MuJoCoSimulation.send_joint_targets`` (same manifest, 112 px ``onboard_rgb``) and compares
realised joint motion, tracking error, contacts, timing and onboard frames. Runs with the v2
apple and plate (``isaac_parity.json`` ``objects: true``) use the v2 MuJoCo scene at the same
seed-0 reset layout; older robot-only runs make the apple and plate invisible and
non-colliding. Frames are scored with the declared metric of the committed camera manifest
(``image_parity.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from embodied_jepa.apple_to_plate_v2 import apply_v2_scene
from embodied_jepa.isaac_scene import canonical_sha256
from embodied_jepa.isaac_transport import (
    joint_manifest_from_mujoco,
    manifest_sha256,
    reset_pose,
)
from embodied_jepa.simulation import MuJoCoSimulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_parity  # noqa: E402
from compare_joints import group  # noqa: E402
from parity_targets import trajectory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FRAMES = (0, 30, 50, 70, 90)


def hide_objects(sim: MuJoCoSimulation) -> None:
    model = sim.model
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name
        if body in ("apple", "plate"):
            model.geom_rgba[g, 3] = 0.0
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0


def robot_contacts(sim: MuJoCoSimulation) -> list[list[str]]:
    """MuJoCo contacts involving the robot: [other geom or body, robot body], sorted, unique."""
    model, data = sim.model, sim.data
    robot = {model.body(i).name for i in range(1, model.nbody)} - {"apple", "plate"}
    out = set()
    for c in data.contact[: data.ncon]:
        g = (int(c.geom1), int(c.geom2))
        bodies = [model.body(int(model.geom_bodyid[x])).name for x in g]
        labels = [model.geom(x).name or b for x, b in zip(g, bodies, strict=True)]
        for i in (0, 1):
            if bodies[i] in robot:
                out.add((labels[1 - i], bodies[i]))
    return [list(p) for p in sorted(out)]


def scene_contacts(sim: MuJoCoSimulation) -> list[dict]:
    """Contacts between distinct bodies, labelled like the Isaac transport's contacts.

    Labels: ``table``, ``floor``, ``apple``, ``plate`` or the robot body name. The force is the
    normal-plus-friction contact force magnitude from ``mj_contactForce`` (N), summed per pair.
    Robot self-contacts are listed too (flag ``self``); Isaac has self-collision off."""
    import mujoco

    model, data = sim.model, sim.data
    pairs: dict[tuple[str, str], float] = {}
    force = np.zeros(6)
    for i, c in enumerate(data.contact[: data.ncon]):
        labels = []
        for g in (int(c.geom1), int(c.geom2)):
            body = model.body(int(model.geom_bodyid[g])).name
            name = model.geom(g).name
            labels.append(name if name in ("table", "floor") else body)
        if labels[0] == labels[1]:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        key = tuple(sorted(labels))
        pairs[key] = pairs.get(key, 0.0) + float(np.linalg.norm(force[:3]))
    scene = {"table", "floor", "apple", "plate"}
    return [
        {"bodies": list(k), "force_n": f, "self": not (set(k) & scene)}
        for k, f in sorted(pairs.items())
    ]


def normalized_joints(manifest: dict) -> list[dict]:
    """Joint rows with the v0 key ``frictionloss_not_applied`` renamed to ``frictionloss``."""
    rows = []
    for j in manifest["joints"]:
        j = dict(j)
        if "frictionloss_not_applied" in j:
            j["frictionloss"] = j.pop("frictionloss_not_applied")
        rows.append(j)
    return rows


def stats(x: np.ndarray) -> dict:
    return {
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True, help="parity_isaac.py output dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="joint manifest the Isaac run used (default: the committed v1 or legacy v0 file "
        "whose hash the run recorded)",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    trace = np.load(args.isaac / "isaac_trace.npz")
    isaac_record = json.loads((args.isaac / "isaac_parity.json").read_text())
    names = [str(n) for n in trace["joint_names"]]
    targets = trace["targets"]
    size = isaac_record["rgb_size"]
    width, height = (size, size) if isinstance(size, int) else size
    objects = bool(isaac_record.get("objects", False))
    variant = isaac_record.get("trajectory", "table_contact_v0")
    camera_manifest = json.loads((ROOT / "configs/isaac/onboard_camera_v1.json").read_text())
    if "camera_manifest_sha256" in isaac_record and isaac_record[
        "camera_manifest_sha256"
    ] != canonical_sha256(camera_manifest):
        raise SystemExit("the Isaac run used a camera manifest that is not the committed one")

    sim = MuJoCoSimulation(width=width, height=height, render=True)
    v2_record = apply_v2_scene(sim.model)
    candidates = (
        [args.manifest]
        if args.manifest
        else [ROOT / f"configs/isaac/g1_dex3_joint_manifest_v{v}.json" for v in (1, 0)]
    )
    used = next(
        (
            p
            for p in candidates
            if manifest_sha256(json.loads(p.read_text())) == isaac_record["manifest_sha256"]
        ),
        None,
    )
    if used is None:
        raise SystemExit("Isaac run used a joint manifest that is not committed")
    committed = json.loads(used.read_text())
    regenerated = joint_manifest_from_mujoco(sim)
    if normalized_joints(committed) != normalized_joints(regenerated) or any(
        committed[k] != regenerated[k] for k in regenerated if k not in ("joints", "version")
    ):
        raise SystemExit("the run's joint manifest differs from the MuJoCo model")
    if list(sim.joint_names) != names:
        raise SystemExit("joint order differs")
    if objects:
        sim.reset(0)  # the seed-0 default layout, as IsaacTransport.reset(0)
    else:
        hide_objects(sim)
        # Objects far from the arms' reach and invisible/non-colliding: robot + table only.
        sim.reset(0, object_xy=[0.63, 0.30], plate_xy=[0.60, -0.24])
    sim.render()  # warm the renderer; the first MuJoCo render is discarded elsewhere too
    if not np.array_equal(reset_pose(committed), sim.data.qpos[sim.qadr]):
        raise SystemExit("manifest reset pose differs from MuJoCoSimulation.reset")
    regenerated_targets = trajectory(
        names, reset_pose(committed), ROOT / "configs/g1_sim_action.json", variant
    )
    targets_equal = bool(np.array_equal(regenerated_targets, targets))

    qpos, qvel, times, step_ms, render_ms = [], [], [], [], []
    frames, classes, apple = {}, {}, []
    obs = sim.read()
    classes[0] = image_parity.mujoco_classes(sim)
    apple.append(sim.task_truth()["object_position"].tolist())
    full_contacts = [scene_contacts(sim)]
    qpos.append(obs["qpos"])
    qvel.append(obs["qvel"])
    times.append(obs["timestamp"])
    frames[0] = obs["rgb"]
    contacts = [robot_contacts(sim)]
    for k, target in enumerate(targets, start=1):
        a = time.perf_counter()
        ack = sim.send_joint_targets(target, joint_names=names, deadline=sim.data.time + 0.05)
        b = time.perf_counter()
        obs = sim.read()
        c = time.perf_counter()
        if ack["status"] != "applied":
            raise RuntimeError(f"interval {k}: {ack}")
        step_ms.append(1e3 * (b - a))
        render_ms.append(1e3 * (c - b))
        qpos.append(obs["qpos"])
        qvel.append(obs["qvel"])
        times.append(obs["timestamp"])
        contacts.append(robot_contacts(sim))
        full_contacts.append(scene_contacts(sim))
        apple.append(sim.task_truth()["object_position"].tolist())
        if k in FRAMES:
            frames[k] = obs["rgb"]
            classes[k] = image_parity.mujoco_classes(sim)
    sim.close()
    mq, mv, mt = np.array(qpos), np.array(qvel), np.array(times)
    iq, it = trace["qpos"][0], trace["time"][0]

    # Joint motion: Isaac vs MuJoCo, and each vs its commanded target (after each interval).
    diff = np.abs(iq - mq)  # [91, 43]
    groups = sorted({group(n) for n in names})
    by_group = {
        g: {
            "isaac_vs_mujoco_rad": stats(diff[:, [group(n) == g for n in names]]),
            "mujoco_tracking_rad": stats(
                np.abs(mq[1:] - targets)[:, [group(n) == g for n in names]]
            ),
            "isaac_tracking_rad": stats(
                np.abs(iq[1:] - targets)[:, [group(n) == g for n in names]]
            ),
        }
        for g in groups
    }
    worst = np.unravel_index(np.argmax(diff), diff.shape)
    # MuJoCo robot contacts per read (Isaac contacts are not measured).
    contact_reads: dict[str, list[int]] = {}
    for k, pairs in enumerate(contacts):
        for other, body in pairs:
            contact_reads.setdefault(f"{body}|{other}", []).append(k)
    link = {n: n.replace("_joint", "_link") for n in names}
    per_joint_worst = {
        n: {
            "max_abs_diff_rad": float(diff[:, i].max()),
            "read_index": int(np.argmax(diff[:, i])),
            "mujoco_contact_on_own_link_at_that_read": any(
                body == link[n] for _, body in contacts[int(np.argmax(diff[:, i]))]
            ),
        }
        for i, n in enumerate(names)
        if diff[:, i].max() > 0.01
    }
    phases = {
        "hold 0-10": (0, 11),
        "arms 10-50": (11, 51),
        "hands 50-70": (51, 71),
        "return 70-90": (71, 91),
    }
    by_phase = {p: float(diff[a:b].max()) for p, (a, b) in phases.items()}
    per_joint_max = {n: float(diff[:, i].max()) for i, n in enumerate(names)}

    from PIL import Image

    args.output.mkdir(parents=True)
    image_rows, repeat_rows = {}, {}
    for k, rgb in frames.items():
        prefix = (
            "isaac_r0_" if (args.isaac / "frames" / f"isaac_r0_{k:03d}.png").exists() else "isaac_"
        )
        isaac_png = args.isaac / "frames" / f"{prefix}{k:03d}.png"
        isaac_rgb = np.asarray(Image.open(isaac_png).convert("RGB"))
        seg_path = args.isaac / "frames" / f"{prefix}{k:03d}_seg.npy"
        isaac_cls = None
        if seg_path.exists():
            labels = json.loads(seg_path.with_suffix(".json").read_text())
            isaac_cls = image_parity.isaac_classes(np.load(seg_path), labels)
        image_rows[k] = image_parity.compare(isaac_rgb, rgb, isaac_cls, classes[k])
        repeat = args.isaac / "frames" / f"isaac_r1_{k:03d}.png"
        if repeat.exists():
            r1 = np.asarray(Image.open(repeat).convert("RGB")).astype(int)
            repeat_rows[k] = int(np.abs(r1 - isaac_rgb.astype(int)).max())
        Image.fromarray(np.concatenate([rgb, isaac_rgb], axis=1)).save(
            args.output / f"mujoco_left_isaac_right_{k:03d}.png"
        )
        if isaac_cls is not None:
            palette = np.array(
                [[40, 40, 40], [230, 230, 230], [150, 110, 70], [220, 30, 20], [40, 90, 220]],
                dtype=np.uint8,
            )
            Image.fromarray(np.concatenate([palette[classes[k]], palette[isaac_cls]], axis=1)).save(
                args.output / f"classes_mujoco_left_isaac_right_{k:03d}.png"
            )
    image_verdict = image_parity.verdict(image_rows, camera_manifest["image_parity"])

    # Contacts per read: robot link or apple against the scene, in both simulators.
    def scene_pairs(rows):
        return sorted(tuple(r["bodies"]) for r in rows if not r.get("self"))

    isaac_last = isaac_record.get("isaac_contacts_last_step_per_read")
    isaac_any = isaac_record.get("isaac_contacts_any_substep_per_read")
    contact_rows = []
    for k in range(len(full_contacts)):
        row = {"read": k, "mujoco": scene_pairs(full_contacts[k])}
        if isaac_last is not None:
            row["isaac_last_step"] = scene_pairs(isaac_last[k])
            row["isaac_any_substep"] = sorted(tuple(p) for p in isaac_any[k])
        contact_rows.append(row)

    def robot_scene(pairs):
        return [
            p
            for p in pairs
            if not ({"apple"} <= set(p) and ({"table"} <= set(p) or {"plate"} <= set(p)))
        ]

    contact_summary = {
        "reads_with_robot_scene_contact_mujoco": [
            r["read"] for r in contact_rows if robot_scene(r["mujoco"])
        ],
        "reads_with_robot_scene_contact_isaac_last_step": [
            r["read"] for r in contact_rows if robot_scene(r.get("isaac_last_step", []))
        ],
        "reads_with_robot_scene_contact_isaac_any_substep": [
            r["read"] for r in contact_rows if robot_scene(r.get("isaac_any_substep", []))
        ],
        "mujoco_robot_self_contact_reads": [
            k for k, rows in enumerate(full_contacts) if any(r.get("self") for r in rows)
        ],
        "apple_pairs_mujoco_final": [p for p in contact_rows[-1]["mujoco"] if "apple" in p],
        "apple_pairs_isaac_final": [
            p for p in contact_rows[-1].get("isaac_last_step", []) if "apple" in p
        ],
    }
    apple_m = np.array(apple, dtype=float)
    apple_i = isaac_record.get("apple_position_per_read")
    apple_rows = None
    if objects and apple_i is not None:
        apple_i = np.array(apple_i, dtype=float)
        apple_rows = {
            "mujoco_first_last": [apple_m[0].tolist(), apple_m[-1].tolist()],
            "isaac_first_last": [apple_i[0].tolist(), apple_i[-1].tolist()],
            "max_abs_diff_m": float(np.abs(apple_m - apple_i).max()),
            "mujoco_max_displacement_m": float(np.abs(apple_m - apple_m[-1]).max()),
            "isaac_max_displacement_m": float(np.abs(apple_i - apple_i[-1]).max()),
        }

    report = {
        "question": "Does IsaacTransport on the converted USD realise the same joint motion "
        "as MuJoCoSimulation for identical joint targets? (robot, floor and table, plus the "
        "v2 apple and plate when the run has objects)",
        "isaac_run": str(args.isaac),
        "manifest_sha256": manifest_sha256(committed),
        "targets_regenerated_equal": targets_equal,
        "intervals": int(targets.shape[0]),
        "episode_time_equal": bool(np.allclose(it, mt, atol=1e-9)),
        "joint_abs_diff_by_group": by_group,
        "joint_abs_diff_max_by_phase_rad": by_phase,
        "worst": {
            "joint": names[int(worst[1])],
            "read_index": int(worst[0]),
            "rad": float(diff[worst]),
        },
        "final_pose_abs_diff_max_rad": float(diff[-1].max()),
        "per_joint_max_abs_diff_rad": per_joint_max,
        "isaac_repeat_max_abs_qpos_diff": isaac_record["repeat_max_abs_qpos_diff"],
        "timing_ms": {
            "mujoco_send_joint_targets": stats(np.array(step_ms)),
            "mujoco_read_with_render": stats(np.array(render_ms)),
            "isaac_send_joint_targets": isaac_record["send_joint_targets_ms"],
            "isaac_read_with_render": isaac_record["read_with_render_ms"],
        },
        "trajectory": variant,
        "objects": objects,
        "mujoco_v2_scene": v2_record,
        "image_parity": image_rows,
        "image_parity_verdict": image_verdict,
        "isaac_image_repeat_max_abs_diff": repeat_rows,
        "isaac_freshness": isaac_record.get("freshness", isaac_record.get("staleness")),
        "contacts_per_read": contact_rows,
        "contact_summary": contact_summary,
        "apple_position": apple_rows,
        "isaac_rejections": isaac_record["rejections"],
        "isaac_after_close": isaac_record["after_close"],
        "mujoco_contacts_by_pair_read_indices": contact_reads,
        "joints_over_0.01_rad": per_joint_worst,
        "isaac_joint_friction": isaac_record.get("joint_friction", "not recorded"),
        "caveats": [
            {
                "frictionloss": "MuJoCo frictionloss applied in Isaac as PhysX static/dynamic "
                "friction effort (closest model, not the same algorithm)",
                "none": "MuJoCo frictionloss (0.1-0.2 N m) not applied in Isaac (run option)",
            }.get(
                isaac_record.get("joint_friction"),
                "Isaac joint friction mode not recorded; this run predates the friction fix "
                "and had PhysX's load-proportional legacy coefficient active",
            ),
            "MuJoCo self-collision is on, Isaac self-collision is off (importer default)",
            "different engines, integrators (implicitfast vs PhysX TGS) and renderers",
            "contacts compared as body pairs at the end of each interval (MuJoCo) and at the "
            "last physics step / any substep of the interval (Isaac); not contact parity",
        ],
    }
    np.savez_compressed(args.output / "mujoco_trace.npz", qpos=mq, qvel=mv, time=mt)
    (args.output / "parity_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "targets_regenerated_equal",
                    "episode_time_equal",
                    "joint_abs_diff_max_by_phase_rad",
                    "worst",
                    "final_pose_abs_diff_max_rad",
                    "isaac_repeat_max_abs_qpos_diff",
                    "timing_ms",
                    "image_parity_verdict",
                    "isaac_image_repeat_max_abs_diff",
                    "isaac_freshness",
                    "contact_summary",
                    "apple_position",
                    "joints_over_0.01_rad",
                )
            },
            indent=2,
        )
    )
    for g, row in by_group.items():
        print(
            f"{g:6s} isaac-vs-mujoco max {row['isaac_vs_mujoco_rad']['max']:.4f} "
            f"median {row['isaac_vs_mujoco_rad']['median']:.4f} | tracking max "
            f"mj {row['mujoco_tracking_rad']['max']:.4f} "
            f"isaac {row['isaac_tracking_rad']['max']:.4f}"
        )


if __name__ == "__main__":
    main()
