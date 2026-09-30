"""Host side of the e9 cross-simulator check, MuJoCo against Isaac/Newton (development).

Not a gated run, not a learned or policy result: e9 is TASK-070's privileged scripted expert
(``docs/experiments/apple_to_plate_v2_expert.md``). Development seeds only
(``isaac_e9.SEEDS``, 50200-50215). Report: ``docs/ISAAC_E9_REPLAY.md``.

    # 1. MuJoCo reference: e9 closed loop, recording every joint-target command
    uv run --no-sync python scripts/isaac/e9_replay.py mujoco --output outputs/<ref>
    # 2. Isaac server in the container (see e9_server_isaac.py), then the client:
    uv run --no-sync python scripts/isaac/e9_replay.py isaac --reference outputs/<ref> \\
        --socket outputs/<srv>/run/e9.sock --output outputs/<isaac>
    #    (--sham runs the same client against a host MuJoCo endpoint: a plumbing check)
    # 3. Per-attempt comparison
    uv run --no-sync python scripts/isaac/e9_replay.py compare --reference outputs/<ref> \\
        --isaac outputs/<isaac> --output outputs/<isaac>/compare
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import apple_to_plate_v2 as v2  # noqa: E402
from embodied_jepa import isaac_e9 as ie  # noqa: E402
from embodied_jepa import resting_expert as rx  # noqa: E402

PAIR = {k: i for i, k in enumerate(rx.PAIR_TYPES)}
IPAIR = {k: i for i, k in enumerate(ie.ISAAC_PAIR_TYPES)}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    return {
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "version": ie.VERSION,
        "expert": ie.E9,
        "scene": v2.SCENE_VERSION,
    }


def load_wide_reset():
    spec = importlib.util.spec_from_file_location(
        "_evaluate_apple_reset", ROOT / "scripts" / "evaluate_apple.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wide_reset


def attempt_plan(seeds, levels) -> list[dict]:
    wide_reset = load_wide_reset()
    plan = []
    for level in levels:
        offsets = rx.plate_offsets(len(v2.DEV_SEEDS), level, v2.DEV_DIRECTION_SEED)
        for seed in seeds:
            r = wide_reset(seed)
            plan.append(
                {
                    "key": f"{level}-{seed}",
                    "seed": int(seed),
                    "plate_cm": float(level),
                    "reset": {"object_xy": list(r["object_xy"]), "plate_xy": list(r["plate_xy"])},
                    "plate_offset": offsets[v2.DEV_SEEDS.index(seed)].tolist(),
                }
            )
    return plan


def closed_loop(robot, recorder, bounds, task) -> tuple[dict, dict]:
    summary, arrays = rx.run_attempt(
        robot,
        bounds,
        seed=task["seed"],
        reset=task["reset"],
        plate_offset=task["plate_offset"],
        make_expert=lambda truth: rx.RestingPlaceExpert(truth, **ie.E9),
    )
    rec = recorder.arrays()
    summary["look_steps"] = int(len(rec["rec_time"]) - summary["steps"])
    summary.update(self_contact_fields(recorder, rec))
    return summary, {**arrays, **rec}


def self_contact_fields(recorder, rec) -> dict:
    """Per-attempt robot self-contact (see ``isaac_e9.self_contact_summary``)."""
    return {
        "self_contact_pairs": dict(recorder.self_pairs),
        "self_contact_steps": int(np.sum(np.asarray(rec["rec_self_count"]) > 0)),
        "remote_self_contact_steps": int(
            np.sum(np.asarray(rec["rec_isaac_counts"])[:, IPAIR["robot_self"]] > 0)
        ),
    }


def warn_self_contact(summary: dict) -> None:
    if summary["flag"]:
        hits = {k: v["attempts_with_contact"] for k, v in summary.items() if k != "flag"}
        print(f"SELF-CONTACT FLAG: robot self-contact seen: {hits}", file=sys.stderr, flush=True)


# ----- MuJoCo reference ----------------------------------------------------------------------
_W: dict = {}


def _mujoco_init() -> None:
    from embodied_jepa import first_policy_runtime as rt

    _W["robot"], _W["scene"] = v2.make_v2_robot()
    _W["recorder"] = ie.StepRecorder(_W["robot"])
    _W["bounds"] = rt.configured_bounds()


def _mujoco_one(task: dict) -> dict:
    out = Path(task["dir"])
    try:
        started = time.monotonic()
        summary, arrays = closed_loop(_W["robot"], _W["recorder"], _W["bounds"], task)
        summary["seconds"] = time.monotonic() - started
        np.savez_compressed(out / f"closed_{task['key']}.npz", **arrays)
        # MuJoCo replaying its own commands open loop must reproduce the closed loop exactly.
        replay, rarrays = ie.replay_open_loop(
            _W["robot"],
            seed=task["seed"],
            reset=task["reset"],
            initial_q=arrays["rec_initial_q"],
            targets=arrays["rec_targets"],
            look_steps=summary["look_steps"],
        )
        rq = _W["recorder"].arrays()["rec_q"]
        replay["max_joint_diff_vs_closed"] = float(np.abs(rq - arrays["rec_q"]).max())
        replay["final_apple_diff_m"] = float(
            np.abs(rarrays["apple_pos"][-1] - arrays["apple_pos"][-1]).max()
        )
        return {"ok": True, **task, "closed": summary, "self_replay": replay}
    except Exception as error:  # noqa: BLE001 - recorded
        return {"ok": False, **task, "error": f"{type(error).__name__}: {error}"}


def cmd_mujoco(args) -> int:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    prov = provenance()  # at the start: the code that runs
    seeds = ie.check_seeds(args.seeds or ie.SEEDS)
    plan = attempt_plan(seeds, args.levels)
    args.output.mkdir(parents=True)
    for task in plan:
        task["dir"] = str(args.output)
    started = time.monotonic()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_mujoco_init)
    try:
        rows = pool.map(_mujoco_one, plan, chunksize=1)
    finally:
        pool.terminate()
        pool.join()
    report = {
        "what": "e9 MuJoCo reference for the Isaac replay (development; privileged scripted "
        "expert; not a gated run; not a learned result)",
        **prov,
        "seeds": list(seeds),
        "levels_cm": list(args.levels),
        "seconds": time.monotonic() - started,
        "attempts": rows,
        "self_contact": ie.self_contact_summary([r for r in rows if r["ok"]], ["closed"]),
    }
    write_report(args.output / "report.json", report)
    warn_self_contact(report["self_contact"])
    ok = [r for r in rows if r["ok"]]
    print(
        json.dumps(
            {
                "attempts": len(rows),
                "errors": len(rows) - len(ok),
                "at_rest": sum(r["closed"]["at_rest"] for r in ok),
                "self_replay_at_rest_equal": sum(
                    r["closed"]["at_rest"] == r["self_replay"]["at_rest"] for r in ok
                ),
                "self_replay_max_joint_diff": max(
                    (r["self_replay"]["max_joint_diff_vs_closed"] for r in ok), default=None
                ),
            },
            indent=1,
        )
    )
    return 0


# ----- Isaac (or sham) client ----------------------------------------------------------------
class SocketEndpoint:
    """The ``MirrorSimulation`` endpoint over the container server's Unix socket."""

    def __init__(self, path: Path, timeout_s: float):
        from multiprocessing.connection import Client

        deadline = time.monotonic() + timeout_s
        key_path = path.parent / "authkey"
        waited = time.monotonic()
        while not (path.exists() and key_path.exists()):
            if time.monotonic() > deadline:
                raise SystemExit(f"no server socket at {path} after {timeout_s:.0f} s")
            time.sleep(2.0)
        self.wait_s = time.monotonic() - waited
        self.conn = Client(str(path), family="AF_UNIX", authkey=bytes.fromhex(key_path.read_text()))
        self.hello = self.call("hello")

    def call(self, cmd: str, **kwargs) -> dict:
        self.conn.send({"cmd": cmd, **kwargs})
        reply = self.conn.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"server {cmd}: {reply.get('error')}\n{reply.get('traceback')}")
        return reply

    def reset(self, **kwargs) -> dict:
        return self.call("reset", **kwargs)["state"]

    def step(self, targets, deadline) -> dict:
        reply = self.call("step", targets=targets, deadline=deadline)
        return {"ack": reply["ack"], "state": reply["state"]}

    def stop(self, reason) -> dict:
        return self.call("stop", reason=reason)["state"]

    def close(self) -> None:
        try:
            self.call("close")
        finally:
            self.conn.close()


def cmd_isaac(args) -> int:
    from embodied_jepa import first_policy_runtime as rt

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    prov = provenance()  # at the start: the code that runs
    ref = json.loads((args.reference / "report.json").read_text())
    tasks = [r for r in ref["attempts"] if r["ok"]]
    if args.only:
        tasks = [r for r in tasks if r["key"] in set(args.only)]
    ie.check_seeds(sorted({t["seed"] for t in tasks}))
    args.output.mkdir(parents=True)
    started = time.monotonic()
    if args.sham:
        endpoint = ie.MuJoCoEndpoint()
        backend = {"endpoint": "host MuJoCo (sham)", "physics": "mujoco"}
    else:
        endpoint = SocketEndpoint(args.socket, args.connect_timeout)
        backend = {
            "endpoint": str(args.socket),
            "physics": endpoint.hello["physics"],
            "socket_wait_s": endpoint.wait_s,
        }
    robot, scene = ie.make_mirror_robot(endpoint, render=False)
    if not args.sham and list(endpoint.hello["joint_names"]) != list(robot.sim.joint_names):
        raise SystemExit("the server's joint order differs from the mirror's")
    recorder = ie.StepRecorder(robot)
    bounds = rt.configured_bounds()
    rows = []
    try:
        for task in tasks:
            ref_arrays = np.load(args.reference / f"closed_{task['key']}.npz")
            row = {k: task[k] for k in ("key", "seed", "plate_cm", "reset", "plate_offset")}
            for mode in args.modes:
                t0 = time.monotonic()
                try:
                    if mode == "closed":
                        summary, arrays = closed_loop(robot, recorder, bounds, task)
                    else:
                        look = task["closed"]["look_steps"]
                        summary, arrays = ie.replay_open_loop(
                            robot,
                            seed=task["seed"],
                            reset=task["reset"],
                            initial_q=ref_arrays["rec_initial_q"],
                            targets=ref_arrays["rec_targets"],
                            look_steps=look,
                        )
                        summary["look_steps"] = look
                        rec = recorder.arrays()
                        summary.update(self_contact_fields(recorder, rec))
                        arrays = {**arrays, **rec}
                    summary["seconds"] = time.monotonic() - t0
                    np.savez_compressed(args.output / f"{mode}_{task['key']}.npz", **arrays)
                    row[mode] = {"ok": True, **summary}
                except Exception as error:  # noqa: BLE001 - recorded, next attempt
                    row[mode] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    robot.stop("error")
                print(
                    task["key"],
                    mode,
                    {
                        k: row[mode].get(k)
                        for k in ("at_rest", "final_distance_cm", "stop_reason", "error")
                    },
                    f"{time.monotonic() - t0:.1f}s",
                    flush=True,
                )
            rows.append(row)
        timing = endpoint.call("info") if not args.sham else None
    finally:
        if not args.keep_server:
            endpoint.close()
    report = {
        "what": "e9 in Isaac through the mirror harness (development; privileged scripted "
        "expert; not a gated run; not a learned result)",
        **prov,
        **backend,
        "reference": str(args.reference),
        "reference_revision": ref["revision"],
        "mirror_scene": scene,
        "modes": list(args.modes),
        "server_timing": timing,
        "seconds": time.monotonic() - started,
        "attempts": rows,
        "self_contact": ie.self_contact_summary(rows, args.modes),
    }
    write_report(args.output / "report.json", report)
    warn_self_contact(report["self_contact"])
    return 0


# ----- MuJoCo sensitivity ------------------------------------------------------------------
def _sensitivity_one(task: dict) -> dict:
    """MuJoCo open-loop replay of the reference commands with the apple's reset xy moved by
    ``offset_m`` and, optionally, seeded noise on the joint targets (a calibration of how much a
    small difference changes the outcome in MuJoCo itself)."""
    ref = np.load(Path(task["reference"]) / f"closed_{task['key']}.npz")
    reset = dict(task["reset"])
    reset["object_xy"] = (np.asarray(reset["object_xy"]) + task["offset_m"]).tolist()
    targets = np.asarray(ref["rec_targets"], float).copy()
    if task["noise_rad"] > 0:
        # seeded Gaussian noise on every non-leg joint target after the look, inside limits
        sim = _W["robot"].sim
        rng = np.random.default_rng(task["noise_seed"])
        look = task["closed"]["look_steps"]
        legs = np.array([any(k in n for k in ("hip", "knee", "ankle")) for n in sim.joint_names])
        noise = rng.normal(0.0, task["noise_rad"], targets[look:].shape)
        noise[:, legs] = 0.0
        limits = sim.model.jnt_range[sim.joint_ids]
        targets[look:] = np.clip(targets[look:] + noise, limits[:, 0], limits[:, 1])
    summary, arrays = ie.replay_open_loop(
        _W["robot"],
        seed=task["seed"],
        reset=reset,
        initial_q=ref["rec_initial_q"],
        targets=targets,
        look_steps=task["closed"]["look_steps"],
    )
    rq = _W["recorder"].arrays()["rec_q"]
    n = min(len(rq), len(ref["rec_q"]))
    return {
        "key": task["key"],
        "offset_m": task["offset_m"],
        "noise_rad": task["noise_rad"],
        "noise_seed": task["noise_seed"],
        "at_rest": summary["at_rest"],
        "reference_at_rest": task["closed"]["at_rest"],
        "final_distance_cm": summary["final_distance_cm"],
        "final_xy_gap_cm": float(
            np.linalg.norm(arrays["apple_pos"][-1][:2] - ref["apple_pos"][-1][:2]) * 100
        ),
        "max_joint_diff": float(np.abs(rq[:n] - ref["rec_q"][:n]).max()),
    }


def cmd_sensitivity(args) -> int:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    prov = provenance()  # at the start: the code that runs
    ref = json.loads((args.reference / "report.json").read_text())
    tasks = []
    for r in ref["attempts"]:
        if not r["ok"]:
            continue
        for angle in args.angles_deg:
            a = np.deg2rad(angle)
            offset = args.offset_mm / 1000 * np.array([np.cos(a), np.sin(a)])
            tasks.append(
                {
                    **r,
                    "reference": str(args.reference),
                    "offset_m": offset.tolist(),
                    "noise_rad": args.target_noise_rad,
                    "noise_seed": int(args.noise_seed + len(tasks)),
                }
            )
    args.output.mkdir(parents=True)
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_mujoco_init)
    try:
        rows = pool.map(_sensitivity_one, tasks, chunksize=1)
    finally:
        pool.terminate()
        pool.join()
    flips = sum(r["at_rest"] != r["reference_at_rest"] for r in rows)
    gaps = [r["final_xy_gap_cm"] for r in rows]
    report = {
        "what": "MuJoCo open-loop sensitivity of e9's outcome to a tiny apple reset offset "
        "(development; not a gated run; not a learned result)",
        **prov,
        "reference": str(args.reference),
        "offset_mm": args.offset_mm,
        "target_noise_rad": args.target_noise_rad,
        "noise_seed": args.noise_seed,
        "angles_deg": list(args.angles_deg),
        "replays": len(rows),
        "at_rest": sum(r["at_rest"] for r in rows),
        "outcome_flips": flips,
        "final_xy_gap_cm_q50_q90_max": [
            float(np.median(gaps)),
            float(np.quantile(gaps, 0.9)),
            float(np.max(gaps)),
        ],
        "rows": rows,
    }
    write_report(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    return 0


# ----- comparison -----------------------------------------------------------------------------
def attempt_events(arrays, summary, *, isaac: bool) -> dict:
    """Events after the look. Hand contact: Isaac's own list for an Isaac run (the truth the
    scorer uses there), MuJoCo's for MuJoCo; plate contact: base or rim (mirror geometry)."""
    look = int(summary["look_steps"])
    counts = np.asarray(arrays["counts"])
    if isaac:
        hand = np.asarray(arrays["rec_isaac_counts"])[look:, IPAIR["apple_hand"]] > 0
    else:
        hand = counts[:, PAIR["apple_hand"]] > 0
    plate = (counts[:, PAIR["apple_plate_base"]] > 0) | (counts[:, PAIR["apple_plate_rim"]] > 0)
    pos = np.asarray(arrays["apple_pos"])
    n = min(len(hand), len(pos))
    return ie.events(
        hand=hand[:n],
        apple_pos=pos[:n],
        apple_lin=np.asarray(arrays["apple_lin"])[:n],
        plate_contact=plate[:n],
        start_z=float(np.asarray(arrays["rec_apple_pos"])[0, 2]),
        open_step=summary.get("open_step"),
    )


def cmd_compare(args) -> int:
    ref = json.loads((args.reference / "report.json").read_text())
    isa = json.loads((args.isaac / "report.json").read_text())
    ref_rows = {r["key"]: r for r in ref["attempts"] if r["ok"]}
    names = None
    table = []
    for row in isa["attempts"]:
        key = row["key"]
        m = ref_rows[key]
        m_arr = dict(np.load(args.reference / f"closed_{key}.npz"))
        if names is None:
            from embodied_jepa.simulation import MuJoCoSimulation

            names = MuJoCoSimulation(render=False).joint_names
        out = {
            "key": key,
            "seed": row["seed"],
            "plate_cm": row["plate_cm"],
            "mujoco": {
                "at_rest": m["closed"]["at_rest"],
                "final_distance_cm": m["closed"]["final_distance_cm"],
                "latched": m["closed"]["latched_success"],
                "events": attempt_events(m_arr, m["closed"], isaac=False),
                "self_contact_steps": int(np.sum(m_arr["rec_self_count"] > 0)),
                "self_contact_pairs": m["closed"]["self_contact_pairs"],
                "final_apple": m_arr["apple_pos"][-1].tolist(),
            },
        }
        for mode in isa["modes"]:
            s = row.get(mode, {})
            if not s.get("ok"):
                out[mode] = {"ok": False, "error": s.get("error")}
                continue
            arr = dict(np.load(args.isaac / f"{mode}_{key}.npz"))
            if mode == "open":
                arr["phase"] = m_arr["phase"][: len(arr["distance"])]
                ev_summary = {"look_steps": s["look_steps"], "open_step": m["closed"]["open_step"]}
            else:
                ev_summary = s
            rec_q = arr["rec_q"]
            hand_agree = arr["rec_mirror_hand"] == arr["rec_remote_hand"]
            out[mode] = {
                "ok": True,
                "at_rest": s["at_rest"],
                "stop_reason": s["stop_reason"],
                "final_distance_cm": s["final_distance_cm"],
                "latched": s.get("latched_success"),
                "events": attempt_events(arr, ev_summary, isaac=True),
                "final_apple": arr["apple_pos"][-1].tolist(),
                "final_apple_xy_vs_mujoco_cm": float(
                    np.linalg.norm(arr["apple_pos"][-1][:2] - m_arr["apple_pos"][-1][:2]) * 100
                ),
                "divergence": ie.divergence(m_arr["rec_q"], rec_q, names),
                "self_contact_steps": int(np.sum(arr["rec_self_count"] > 0)),
                "mirror_self_contact_steps": int(np.sum(arr["rec_self_count"] > 0)),
                "mirror_self_min_dist_m": float(np.nanmin(arr["rec_self_min_dist"]))
                if np.any(arr["rec_self_count"] > 0)
                else None,
                "mirror_self_contact_pairs": s.get("self_contact_pairs"),
                "hand_contact_agreement": float(hand_agree.mean()),
                "isaac_self_contact_steps": int(
                    np.sum(np.asarray(arr["rec_isaac_counts"])[:, IPAIR["robot_self"]] > 0)
                ),
            }
            out[mode]["remote_self_contact_steps"] = out[mode]["isaac_self_contact_steps"]
        table.append(out)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "what": "e9 MuJoCo vs Isaac comparison (development; not a gated run; not learned)",
        "reference": str(args.reference),
        "isaac": str(args.isaac),
        "reference_revision": ref["revision"],
        "isaac_revision": isa["revision"],
        "physics": isa.get("physics"),
        "compare_revision": git("rev-parse", "HEAD"),
        "attempts": table,
        "totals": totals(table, isa["modes"]),
        "self_contact": ie.self_contact_summary(table, ["mujoco", *isa["modes"]]),
    }
    write_report(args.output / "compare.json", result)
    (args.output / "compare.md").write_text(markdown(table, isa["modes"]))
    print(json.dumps(result["totals"], indent=1))
    print(json.dumps({"self_contact": result["self_contact"]}, indent=1))
    warn_self_contact(result["self_contact"])
    return 0


def cmd_repeat(args) -> int:
    """Two Isaac runs of the same attempts (separate processes): how repeatable is Isaac?"""
    a = json.loads((args.a / "report.json").read_text())
    b = json.loads((args.b / "report.json").read_text())
    rows_b = {r["key"]: r for r in b["attempts"]}
    out = []
    for ra in a["attempts"]:
        rb = rows_b.get(ra["key"])
        if rb is None:
            continue
        for mode in sorted(set(a["modes"]) & set(b["modes"])):
            sa, sb = ra.get(mode, {}), rb.get(mode, {})
            if not (sa.get("ok") and sb.get("ok")):
                continue
            xa = np.load(args.a / f"{mode}_{ra['key']}.npz")
            xb = np.load(args.b / f"{mode}_{ra['key']}.npz")
            n = min(len(xa["rec_q"]), len(xb["rec_q"]))
            out.append(
                {
                    "key": ra["key"],
                    "mode": mode,
                    "at_rest_a": sa["at_rest"],
                    "at_rest_b": sb["at_rest"],
                    "final_distance_cm_a": sa["final_distance_cm"],
                    "final_distance_cm_b": sb["final_distance_cm"],
                    "final_xy_gap_cm": float(
                        np.linalg.norm(xa["apple_pos"][-1][:2] - xb["apple_pos"][-1][:2]) * 100
                    ),
                    "max_joint_diff": float(np.abs(xa["rec_q"][:n] - xb["rec_q"][:n]).max()),
                }
            )
    summary = {}
    for mode in sorted({r["mode"] for r in out}):
        rows = [r for r in out if r["mode"] == mode]
        gaps = [r["final_xy_gap_cm"] for r in rows]
        summary[mode] = {
            "attempts": len(rows),
            "at_rest_a": sum(r["at_rest_a"] for r in rows),
            "at_rest_b": sum(r["at_rest_b"] for r in rows),
            "outcome_flips": sum(r["at_rest_a"] != r["at_rest_b"] for r in rows),
            "final_xy_gap_cm_q50_q90_max": [
                float(np.median(gaps)),
                float(np.quantile(gaps, 0.9)),
                float(np.max(gaps)),
            ],
            "max_joint_diff": max(r["max_joint_diff"] for r in rows),
        }
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(
        args.output / "repeat.json",
        {
            "what": "Isaac run-to-run repeatability of e9 (development; not a gated run)",
            "a": str(args.a),
            "b": str(args.b),
            "revision": git("rev-parse", "HEAD"),
            "summary": summary,
            "rows": out,
        },
    )
    print(json.dumps(summary, indent=1))
    return 0


def totals(table, modes) -> dict:
    out = {}
    for level in sorted({r["plate_cm"] for r in table}):
        rows = [r for r in table if r["plate_cm"] == level]
        cell = {"attempts": len(rows), "mujoco_at_rest": sum(r["mujoco"]["at_rest"] for r in rows)}
        for mode in modes:
            ok = [r for r in rows if r[mode]["ok"]]
            cell[f"{mode}_errors"] = len(rows) - len(ok)
            cell[f"{mode}_at_rest"] = sum(r[mode]["at_rest"] for r in ok)
            cell[f"{mode}_agree_with_mujoco"] = sum(
                r[mode]["at_rest"] == r["mujoco"]["at_rest"] for r in ok
            )
            gaps = [r[mode]["final_apple_xy_vs_mujoco_cm"] for r in ok]
            cell[f"{mode}_final_xy_gap_cm_q50_max"] = (
                [float(np.median(gaps)), float(np.max(gaps))] if gaps else None
            )
        out[str(level)] = cell
    return out


def _fmt(v, digits=2):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def markdown(table, modes) -> str:
    lines = []
    head = "| attempt | MuJoCo at rest / final cm |"
    sep = "|---|---|"
    for mode in modes:
        head += f" {mode}: at rest / final cm / xy gap cm / joint max rad (arms, hands) |"
        sep += "---|"
    lines += [head, sep]
    for r in table:
        line = (
            f"| {r['key']} | {_fmt(r['mujoco']['at_rest'])} / "
            f"{_fmt(r['mujoco']['final_distance_cm'])} |"
        )
        for mode in modes:
            s = r[mode]
            if not s["ok"]:
                line += f" error: {s['error']} |"
                continue
            g = s["divergence"]["groups"]
            line += (
                f" {_fmt(s['at_rest'])} / {_fmt(s['final_distance_cm'])} / "
                f"{_fmt(s['final_apple_xy_vs_mujoco_cm'])} / "
                f"{g['arms']['max']:.3g}, {g['hands']['max']:.3g} |"
            )
        lines.append(line)
    lines += [
        "",
        "Events (steps after the look): grasp / lift / lifted / release / landing / settle",
        "",
        "| attempt | MuJoCo | " + " | ".join(modes) + " |",
        "|---|---|" + "---|" * len(modes),
    ]
    keys = ("grasp_step", "lift_step", "lifted_step", "release_step", "landing_step", "settle_step")
    for r in table:
        cells = [" / ".join(_fmt(r["mujoco"]["events"][k]) for k in keys)]
        for mode in modes:
            s = r[mode]
            cells.append(" / ".join(_fmt(s["events"][k]) for k in keys) if s["ok"] else "error")
        lines.append(f"| {r['key']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: Path, report: dict) -> None:
    payload = json.dumps(report, indent=1, sort_keys=True, default=float) + "\n"
    path.write_text(payload)
    print(path, "sha256", hashlib.sha256(payload.encode()).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mujoco")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--levels", type=float, nargs="+", default=list(ie.LEVELS_CM))
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("isaac")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--socket", type=Path)
    p.add_argument("--sham", action="store_true", help="host MuJoCo endpoint (plumbing check)")
    p.add_argument("--modes", nargs="+", choices=("closed", "open"), default=["closed", "open"])
    p.add_argument("--only", nargs="*", help="attempt keys, e.g. 0.0-50200 (development)")
    p.add_argument("--connect_timeout", type=float, default=1800.0)
    p.add_argument("--keep_server", action="store_true")
    p = sub.add_parser("sensitivity")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--offset_mm", type=float, default=0.1)
    p.add_argument("--angles_deg", type=float, nargs="+", default=[0.0, 90.0, 180.0, 270.0])
    p.add_argument("--target_noise_rad", type=float, default=0.0)
    p.add_argument("--noise_seed", type=int, default=9025)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("repeat")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--isaac", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "isaac" and not args.sham and args.socket is None:
        parser.error("isaac needs --socket or --sham")
    return {
        "mujoco": cmd_mujoco,
        "isaac": cmd_isaac,
        "sensitivity": cmd_sensitivity,
        "repeat": cmd_repeat,
        "compare": cmd_compare,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
