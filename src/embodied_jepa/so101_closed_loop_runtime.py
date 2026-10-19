"""TASK-101's runtime (protocol ``docs/experiments/so101_closed_loop_lift_v1.md`` §3-§6, §12):
the live featuriser, the J/N planner-model adapter with the last and mean cost forms and the
scene-blind substitution, the chained-preview projector, the controllers, the goal demonstration,
the episode loop and the worker pool.

NumPy at import; torch, MuJoCo and the TASK-100 model are imported inside the functions that need
them. Nothing in ``import embodied_jepa`` loads this module.

What each controller reads (§4): the current ``wrist_rgb`` and ``overview_rgb`` frames, its own
measured joint positions, its own joint-position targets (the command history) and the goal
(subgoal latents and subgoal joint positions). No controller object holds the simulator; while a
controller acts, the runner's sentinel makes every simulator-truth accessor raise.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[2]


def shared_root() -> Path:
    """The main checkout (outputs/, checkpoints/ and data/ live behind its symlinks)."""
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (ROOT / common).resolve().parent


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def tree_dirty() -> bool:
    return bool(git("status", "--porcelain", "--untracked-files=no"))


def header(kind: str, **extra) -> dict:
    from embodied_jepa import run_tools as rt

    report = {
        "task": cl.TASK,
        "protocol": cl.PROTOCOL,
        "kind": kind,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": tree_dirty(),
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    } | extra
    rt.assert_local_import(ROOT, report)
    return report


def check_frozen_task100(report: dict) -> None:
    """§12: TASK-100's ``frozen_files_check`` passes (its pinned files are byte-identical)."""
    from embodied_jepa import so101_wm_runtime as wr

    check = wr.frozen_files_check()
    report["task100_frozen_files_ok"] = all(check.values())
    if not all(check.values()):
        raise cl.GuardError(
            f"G-frozen: TASK-100 files differ: {[k for k, v in check.items() if not v]}"
        )


def check_manifest(report: dict, strict: bool) -> None:
    """Gated jobs refuse unless this task's manifest is FROZEN and every pin matches."""
    import hashlib

    path = ROOT / cl.MANIFEST
    if not path.exists():
        if strict:
            raise cl.GuardError("G-frozen: no manifest")
        report["manifest"] = None
        return
    pins = json.loads(path.read_text())
    bad = [
        rel
        for rel, pin in pins.get("files_sha256", {}).items()
        if not (ROOT / rel).exists() or hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != pin
    ]
    report["manifest"] = {"status": pins.get("status"), "files_differing": bad}
    if strict and (pins.get("status") != "FROZEN" or bad):
        raise cl.GuardError(f"G-frozen: status {pins.get('status')}, differing {bad}")


def write_json(path, payload) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=1, default=cl._json_default))
    return cl.sha256_file(path)


# ----- the live latent (§3) --
def load_moments(features_dir=None):
    """TASK-100's fit-sample moments, refused on a fit-report or moments hash mismatch."""
    from embodied_jepa import so101_wm_runtime as wr

    folder = Path(features_dir or shared_root() / cl.TASK100_FEATURES)
    moments, fit = wr.load_fit(folder, cl.FIT_REPORT_SHA256)
    if moments.sha256 != cl.MOMENTS_SHA256:
        raise cl.GuardError("G-moments: the fit moments differ from TASK-100's")
    return moments


def standardise(moments, raw16) -> np.ndarray:
    """float16 [N, 32, 384] -> standardised float32 [N, 12 288]."""
    raw = np.asarray(raw16, np.float16).astype(np.float32).reshape(len(raw16), -1)
    return moments.apply(raw)


class LiveFeaturiser:
    """TASK-100's g4 pipeline: both 224 px frames through the pinned frozen DINOv2, final patch
    tokens pooled to 4 x 4, wrist then overview, rounded to float16 as the store was, then
    standardised with the frozen fit-sample moments (fit report hash checked)."""

    def __init__(self, device: str, features_dir=None):
        from embodied_jepa import so101_wm_data as sd

        self.moments = load_moments(features_dir)
        self.fit_report_sha256 = cl.FIT_REPORT_SHA256
        self.featuriser = sd.Featuriser("pretrained", device=device, batch=64)
        if self.featuriser.digest != cl.ENCODER_DIGEST:
            raise cl.GuardError("G-weights: the DINOv2 weights digest differs from TASK-100's")

    def raw16(self, frames: dict) -> np.ndarray:
        """``{camera: uint8 [N, 224, 224, 3]}`` -> float16 [N, 32, 384].

        TASK-100's featuriser runs a fixed batch of 64 frames per camera; GPU arithmetic depends
        on the batch size, so every call is padded to whole batches of 64 (the last frame
        repeated) and the first N rows are kept. With it, a frame encoded live is bit-identical
        to the stored latent of a frame the store encoded in a full batch (Stage 0 record,
        §7.2.1's VOID and its fix)."""
        n = len(np.asarray(frames[cl.CAMERAS[0]]))
        batch = self.featuriser.batch
        padded = {}
        for c in cl.CAMERAS:
            x = np.asarray(frames[c])
            if len(x) != n:
                raise ContractError("both cameras need the same number of frames")
            extra = (-n) % batch
            padded[c] = np.concatenate([x, np.repeat(x[-1:], extra, 0)]) if extra else x
        pooled = self.featuriser.pooled(padded, ("g4",))
        return pooled["g4"][:n].astype(np.float16)

    def standardise(self, raw16) -> np.ndarray:
        return standardise(self.moments, raw16)

    def encode(self, frames: dict) -> tuple[np.ndarray, np.ndarray]:
        r = self.raw16(frames)
        return r, self.standardise(r)


# ----- models (§3) --
def load_model(name: str, device: str, checkpoints=None):
    """TASK-100's kept checkpoint, refused on a file-hash or implementation-hash mismatch."""
    from embodied_jepa import so101_wm_runtime as wr

    if name not in cl.CHECKPOINT_SHA256:
        raise ContractError(f"no pinned checkpoint {name!r}")
    job = Path(checkpoints or shared_root() / cl.TASK100_CHECKPOINTS) / name
    got = cl.sha256_file(job / "model.pt")
    if got != cl.CHECKPOINT_SHA256[name]:
        raise cl.GuardError(
            f"G-checkpoint: {name}/model.pt sha256 {got[:12]} is not the pinned one"
        )
    model, report = wr.load_job(job, device)
    if model.implementation_sha256 != cl.IMPLEMENTATION_SHA256:
        raise cl.GuardError(f"G-checkpoint: {name}'s implementation hash differs")
    return model, report


class LatentCost:
    """The planner-model adapter (§5.3): ``predict`` rolls the model out on every (projected)
    candidate from the current latent, on the model's device, in the chunks ``predict_features``
    uses; ``distance`` is TASK-100's latent error (mean squared difference over the 12 288
    standardised values, float64) per step, or its running mean over the horizon (``mean``: the
    planner reads the last step, so it sees the step-averaged cost; TASK-089's wrapper).

    ``action_blind`` (N): every candidate gives the model the same inputs (the encoded frame and
    zero commands), so the model is rolled out once and that roll-out is broadcast to every
    candidate; the costs are bit-identical by construction (§6)."""

    def __init__(self, model, cost: str, *, action_blind: bool = False):
        if cost not in cl.COSTS:
            raise ContractError(f"cost must be one of {cl.COSTS}")
        self.model, self.cost, self.action_blind = model, cost, bool(action_blind)
        self.chunk = int(model.config["candidate_chunk_size"])

    def predict(self, latent, candidates):
        import torch

        from embodied_jepa.contracts import validate_actions

        m = self.model
        c = np.asarray(candidates)
        if c.ndim != 4 or c.shape[0] != 1 or c.shape[-1] != cl.JOINTS:
            raise ContractError("candidates must be [1, K, H, 6]")
        validate_actions(c, schema=m.action_schema)
        if c.shape[2] > m.config["max_horizon"]:
            raise ContractError("rollout exceeds max_horizon")
        z = np.asarray(latent, np.float32).reshape(1, -1)
        m.eval()
        with torch.no_grad(), m.rng_scope():
            initial = m._normalize(z)
            if self.action_blind:
                zeros = torch.zeros((1, c.shape[2], cl.JOINTS), dtype=torch.float32)
                out = m.rollout(initial, zeros.to(m.device_name))
                return {"pred": out, "broadcast": c.shape[1]}
            parts = []
            for lo in range(0, c.shape[1], self.chunk):
                part = torch.from_numpy(np.array(c[0, lo : lo + self.chunk], copy=True))
                n = part.shape[0]
                start = initial.expand(n, -1).contiguous()
                parts.append(m.rollout(start, part.to(m.device_name)))
            return {"pred": torch.cat(parts), "broadcast": None}

    def step_errors(self, predictions, goal) -> np.ndarray:
        """float64 [K, H] latent errors."""
        import torch

        pred = predictions["pred"]
        g = torch.as_tensor(np.asarray(goal, np.float32).reshape(-1), device=pred.device).double()
        with torch.no_grad():
            d = pred.double() - g
            err = (d * d).mean(-1)
        if not torch.isfinite(err).all():
            raise cl.GuardError("G-finite: a non-finite latent cost")
        return err.cpu().numpy()

    def distance(self, predictions, goal) -> np.ndarray:
        err = self.step_errors(predictions, goal)
        if self.cost == "mean":
            err = np.cumsum(err, axis=1) / np.arange(1, err.shape[1] + 1)
        out = err.astype(np.float32)
        if predictions["broadcast"] is not None:
            out = np.broadcast_to(out, (predictions["broadcast"], out.shape[1]))
        return np.ascontiguousarray(out)[None]


@dataclass(frozen=True)
class SchemaCandidateProjection:
    """``constraints.CandidateProjection``'s value object (actions [B,K,T,A], feasible bool[B,K],
    read-only copies) validated against ``joint_delta_so101_v0``: the pinned class validates
    against the 14-D ``ee_delta_grasp_v0`` only, and ``SchemaCEMPlanner`` reads just the two
    fields."""

    actions: np.ndarray
    feasible: np.ndarray

    def __post_init__(self):
        from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
        from embodied_jepa.contracts import validate_actions

        validate_actions(self.actions, ndim=4, schema=JOINT_DELTA_SO101_V0)
        if self.feasible.dtype != np.bool_ or self.feasible.shape != self.actions.shape[:2]:
            raise ContractError("candidate feasibility must be bool[B,K]")
        for name in ("actions", "feasible"):
            value = getattr(self, name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class Projector:
    """§5.2: the embodiment's own ``preview``, chained from the current targets (a
    ``constraints.CandidateProjector``); every candidate is feasible."""

    def __init__(self, params: cl.PreviewParams, targets):
        self.params = params
        self.targets = np.asarray(targets, np.float64).copy()

    def __call__(self, requested):
        applied, _ = cl.project_sequences(
            np.asarray(requested, np.float32), self.targets, self.params
        )
        return SchemaCandidateProjection(applied, np.ones(applied.shape[:2], dtype=bool))


# ----- controllers (§5, §6) --
@dataclass
class Goal:
    latents: np.ndarray  # standardised float32 [4, 12288]
    joints: np.ndarray  # float64 [4, 6]


class CEMController:
    """W (and its controls): ``SchemaCEMPlanner`` unchanged, replanned every command, the first
    action of the best candidate executed. It reads frames, measured joints, its own targets and
    the goal."""

    def __init__(
        self,
        *,
        adapter: LatentCost,
        featuriser: LiveFeaturiser,
        horizon: int,
        goal: Goal,
        params: cl.PreviewParams,
        seed: int,
        scene_blind: bool = False,
        single_goal: bool = False,
    ):
        from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
        from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner

        self.adapter, self.featuriser, self.goal, self.params = adapter, featuriser, goal, params
        self.scene_blind = bool(scene_blind)
        self.planner = SchemaCEMPlanner(
            SchemaCEMConfig(
                action_schema=JOINT_DELTA_SO101_V0,
                horizon=int(horizon),
                samples=cl.CEM["samples"],
                iterations=cl.CEM["iterations"],
                elites=cl.CEM["elites"],
                seed=int(seed),
                action_penalty=cl.CEM["action_penalty"],
                minimum_std=cl.CEM["minimum_std"],
                project_candidates=True,
            )
        )
        self.switcher = cl.SubgoalSwitcher(goal.joints, single_goal=single_goal)

    def act(self, frames: dict, q, targets) -> tuple[np.ndarray, dict]:
        import torch

        k = self.switcher.update(q)
        raw16, z = self.featuriser.encode({c: frames[c][None] for c in cl.CAMERAS})
        z = z[0]
        latent = np.zeros_like(z) if self.scene_blind else z
        t0 = time.perf_counter()
        plan = self.planner.plan(
            self.adapter, latent, self.goal.latents[k], projector=Projector(self.params, targets)
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        requested = np.asarray(plan.requested_actions[0, 0], np.float32).copy()
        projected = np.asarray(plan.actions[0, 0], np.float32).copy()
        if not np.isfinite(plan.costs).all():
            raise cl.GuardError("G-finite: a non-finite planned cost")
        return requested, {
            "subgoal": k,
            "projected": projected,
            "cost": float(plan.costs[0]),
            "plan_seconds": seconds,
            "raw16": raw16[0],
        }


class RandController:
    def __init__(self, seed: int):
        self.rng = cl.rand_rng(seed)

    def act(self, frames, q, targets):
        return cl.rand_action(self.rng), {"subgoal": -1}


class HoldController:
    """``SO101Embodiment.hold_action()`` every command (it reads the command targets only)."""

    def __init__(self, hold_action):
        self.hold_action = hold_action

    def act(self, frames, q, targets):
        return np.asarray(self.hold_action(), np.float32), {"subgoal": -1}


class PoseController:
    """H-pose: no image, no model; §5.3's switching and the follower's pose law (§6)."""

    def __init__(self, goal_joints, scales):
        self.goal = np.asarray(goal_joints, np.float64)
        self.scales = np.asarray(scales, np.float32)
        self.switcher = cl.SubgoalSwitcher(self.goal)

    def act(self, frames, q, targets):
        k = self.switcher.update(q)
        return cl.pose_action(targets, self.goal[k], self.scales), {"subgoal": k}


# ----- the simulator, truth sentinel and recorder --
TRUTH_ACCESSORS = ("task_truth", "gripper_contacts")


class TruthSentinel:
    """While ``active``, every simulator-truth accessor raises (§4: no controller reads truth)."""

    def __init__(self, sim):
        self.sim, self.active = sim, False
        for name in TRUTH_ACCESSORS:
            original = getattr(sim, name)

            def guarded(*args, _original=original, _name=name, **kwargs):
                if self.active:
                    raise cl.GuardError(f"G-truth: a controller called sim.{_name}")
                return _original(*args, **kwargs)

            setattr(sim, name, guarded)

    @contextmanager
    def controller(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False


def make_robot(render: bool):
    """A fresh ``SO101Simulation`` (224 px) and embodiment per episode. Without rendering the
    transport's renderer is replaced by an 8 x 8 black frame (S2's ``measure`` precedent)."""
    from embodied_jepa.so101 import SO101Embodiment
    from embodied_jepa.so101_simulation import SO101Simulation

    sim = SO101Simulation(width=cl.IMAGE_SIZE, height=cl.IMAGE_SIZE, render=bool(render))
    if not render:
        sim.render = lambda camera="wrist_rgb": np.zeros((8, 8, 3), np.uint8)
    return SO101Embodiment(sim)


class Recorder:
    """Proxies an embodiment and records every ``execute`` (the goal demonstration and F)."""

    def __init__(self, robot, on_command=None):
        self._robot = robot
        self.on_command = on_command
        self.applied: list[np.ndarray] = []
        self.statuses: list[str] = []

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def execute(self, action):
        result = self._robot.execute(action)
        self.statuses.append(result.status)
        applied = result.applied_action
        self.applied.append(
            np.full(cl.JOINTS, np.nan, np.float32) if applied is None else np.asarray(applied)
        )
        if self.on_command is not None:
            self.on_command(len(self.applied), result)
        return result


def truth_row(sim) -> dict:
    t = sim.task_truth()
    return {
        "cube": np.asarray(t["cube_position"], np.float64),
        "quat": np.asarray(t["cube_quaternion"], np.float64),
        "site": np.asarray(t["gripper_site_position"], np.float64),
        "site_xmat": sim.data.site("gripperframe").xmat.copy().astype(np.float64),
        "contacts": np.array(
            [t["fixed_jaw_contact"], t["moving_jaw_contact"], t["gripper_contact"]], bool
        ),
    }


# ----- the goal demonstration (§4) --
def demonstrate(seed: int, *, render: bool = True) -> dict:
    """Follower v1 (no offset) on ``seed``'s declared reset in its own simulation; captures both
    frames, the measured joints and (for reports only) the cube and fingertip truth after
    commands 40, 70, 90 and 120."""
    from embodied_jepa.so101_followers import follow_lift
    from embodied_jepa.so101_tasks import SO101CubeLiftTask

    robot = make_robot(render)
    task = SO101CubeLiftTask(robot)
    captured = []

    def on_command(n, result):
        if n in cl.SUBGOAL_COMMANDS and result.status != "stopped":
            sim = robot.sim
            frames = (
                np.stack([sim.render(c) for c in cl.CAMERAS])
                if render
                else np.zeros((2, cl.IMAGE_SIZE, cl.IMAGE_SIZE, 3), np.uint8)
            )
            row = truth_row(sim)
            captured.append(
                {
                    "command": n,
                    "frames": frames,
                    "joints": sim.data.qpos[sim.qadr].copy(),
                    "cube_cm": row["cube"] * 100.0,
                    "site": row["site"],
                    "site_xmat": row["site_xmat"],
                }
            )

    rec = Recorder(robot, on_command)
    stopped = False
    try:
        record = follow_lift(rec, task, int(seed))
    except RuntimeError as error:  # the follower raises on a stopped execution
        stopped = True
        record = {"success": False, "termination_reason": f"stopped: {error}"}
    finally:
        robot.close()
    out = {
        "seed": int(seed),
        "success": bool(record.get("success")),
        "termination_reason": record.get("termination_reason"),
        "success_step": (record.get("score") or {}).get("success_step"),
        "stopped": stopped or "stopped" in rec.statuses,
        "commands": len(rec.applied),
        "captured": captured,
        "applied": np.asarray(rec.applied, np.float32).reshape(-1, cl.JOINTS),
    }
    ok, reason = cl.demo_verdict(out)
    out["kept"], out["reason"] = ok, reason
    return out


# ----- the episode loop (§2, §6) --
STATUS_CODES = {"applied": 0, "clipped": 1, "stopped": 2}


def run_episode(task_spec: dict, ctx: dict) -> dict:
    """One arm on one reset. ``task_spec``: arm, seed, slot, config, keep_frames, output.
    ``ctx``: the worker's goal store, featuriser and adapters. Writes ``<output>/<seed>.npz`` (per
    step logs; frames only when ``keep_frames``) and returns the episode summary."""
    from embodied_jepa.so101_tasks import SO101CubeLiftTask

    arm, seed, slot = task_spec["arm"], int(task_spec["seed"]), int(task_spec["slot"])
    cl.check_arm(arm)
    out_dir = Path(task_spec["output"])
    path = out_dir / f"{seed}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    keep = bool(task_spec.get("keep_frames"))
    cem = arm in cl.CEM_ARMS or arm == cl.DIAGNOSTIC
    render = cem or keep
    goals = ctx["goals"]
    started = time.perf_counter()
    if arm == "F":
        return run_follower(seed, slot, goals, path, render=keep, started=started)
    robot = make_robot(render)
    task = SO101CubeLiftTask(robot)
    task.reset(seed)
    sentinel = TruthSentinel(robot.sim)
    params = cl.PreviewParams.of(robot)
    controller = make_controller(
        arm, seed, slot, task_spec.get("config"), goals, ctx, robot, params
    )
    rows: dict[str, list] = {
        k: []
        for k in (
            "subgoal",
            "requested",
            "applied",
            "projected",
            "status",
            "cost",
            "plan_seconds",
            "q",
            "qvel",
            "targets",
            "cube",
            "quat",
            "site",
            "site_xmat",
            "contacts",
            "step_seconds",
        )
    }
    latents, frames, reasons = [], [], []
    termination, score = "running", None
    try:
        for _ in range(cl.MAX_COMMANDS + 1):
            t_step = time.perf_counter()
            obs = robot.observe()
            imgs = {c: obs.images[c][0] for c in cl.CAMERAS}
            q = np.asarray(obs.state.values[0, : cl.JOINTS], np.float64)
            qvel = np.asarray(obs.state.values[0, cl.JOINTS :], np.float64)
            targets = robot.sim.targets.copy()
            if keep:
                frames.append(np.stack([imgs[c] for c in cl.CAMERAS]))
            with sentinel.controller():
                action, info = controller.act(imgs, q, targets)
            if "raw16" in info:
                latents.append(info["raw16"])
            result = robot.execute(np.asarray(action, np.float32))
            if result.status != "stopped" and "projected" in info:
                if not np.array_equal(result.applied_action, info["projected"]):
                    raise cl.GuardError(
                        "G-projection: the executed action differs from the scored one"
                    )
            truth = truth_row(robot.sim)
            rows["subgoal"].append(info.get("subgoal", -1))
            rows["requested"].append(np.asarray(action, np.float32))
            rows["applied"].append(
                np.full(6, np.nan, np.float32)
                if result.applied_action is None
                else np.asarray(result.applied_action, np.float32)
            )
            rows["projected"].append(info.get("projected", np.full(6, np.nan, np.float32)))
            rows["status"].append(STATUS_CODES[result.status])
            rows["cost"].append(info.get("cost", np.nan))
            rows["plan_seconds"].append(info.get("plan_seconds", np.nan))
            rows["q"].append(q)
            rows["qvel"].append(qvel)
            rows["targets"].append(targets)
            for key in ("cube", "quat", "site", "site_xmat", "contacts"):
                rows[key].append(truth[key])
            rows["step_seconds"].append(time.perf_counter() - t_step)
            if result.reason:
                reasons.append(result.reason)
            if result.status == "stopped":
                termination = f"stopped: {result.reason}"
                break
            score = task.evaluate()
            if score["done"]:
                termination = score["termination_reason"]
                break
        else:  # pragma: no cover -- the task's timeout ends every episode first
            raise cl.GuardError("G-steps: the episode outran the task's command budget")
        if cem or keep:  # the final observation, for the reported model-in-the-loop measure
            obs = robot.observe()
            imgs = {c: obs.images[c][0] for c in cl.CAMERAS}
            if keep:
                frames.append(np.stack([imgs[c] for c in cl.CAMERAS]))
            if cem:
                latents.append(ctx["featuriser"].raw16({c: imgs[c][None] for c in cl.CAMERAS})[0])
    finally:
        robot.close()
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    if latents:
        arrays["latents_raw16"] = np.asarray(latents, np.float16)
    if frames:
        arrays["frames"] = np.asarray(frames, np.uint8)
    np.savez_compressed(path, **arrays) if frames else np.savez(path, **arrays)
    success = bool(score and score["success"])
    events = getattr(getattr(controller, "switcher", None), "events", [])
    return {
        "arm": arm,
        "seed": seed,
        "slot": slot,
        "config": task_spec.get("config"),
        "success": success,
        "termination": termination,
        "success_step": score["success_step"] if score else None,
        "max_lift_m": score["max_lift_m"] if score else None,
        "longest_hold_steps": score["longest_hold_steps"] if score else None,
        "commands": len(rows["status"]),
        "stopped": termination.startswith("stopped"),
        "clipped_commands": int(np.sum(arrays["status"] == 1)),
        "reasons": sorted(set(reasons)),
        "switch_events": events,
        "plan_seconds_median": float(np.nanmedian(arrays["plan_seconds"])) if cem else None,
        "plan_seconds_p95": float(np.nanpercentile(arrays["plan_seconds"], 95)) if cem else None,
        "plan_seconds_max": float(np.nanmax(arrays["plan_seconds"])) if cem else None,
        "step_seconds_max": float(np.max(arrays["step_seconds"])),
        "episode_seconds": time.perf_counter() - started,
        "file": path.name,
        "file_sha256": cl.sha256_file(path),
        "frames_kept": bool(frames),
        "pid": os.getpid(),
    }


def make_controller(arm, seed, slot, config, goals, ctx, robot, params):
    a = goals["arrays"]
    joints = np.asarray(a["joints"][slot], np.float64)
    if arm == "rand":
        return RandController(seed)
    if arm == "hold":
        return HoldController(robot.hold_action)
    if arm == "H-pose":
        return PoseController(joints, robot.scales)
    h, cost = cl.parse_config(config)
    goal_slot = int(ctx["derangement"][slot]) if arm == "W-shuf" else slot
    if int(a["seeds"][slot]) != seed:
        raise cl.GuardError("G-goals: the slot's goal belongs to another reset")
    goal = Goal(ctx["goal_latents"][goal_slot], np.asarray(a["joints"][goal_slot], np.float64))
    model_name = cl.ARM_MODEL[arm]
    adapter = ctx["adapter"](model_name, cost)
    return CEMController(
        adapter=adapter,
        featuriser=ctx["featuriser"],
        horizon=h,
        goal=goal,
        params=params,
        seed=cl.planner_seed(seed),
        scene_blind=arm == "W-mean",
        single_goal=arm == cl.DIAGNOSTIC,
    )


def run_follower(seed, slot, goals, path, *, render: bool, started: float) -> dict:
    """F: follower v1 replays the goal demonstration; its executed actions and outcome must equal
    the demonstration's on every kept reset (a guard, §6)."""
    from embodied_jepa.so101_followers import follow_lift
    from embodied_jepa.so101_tasks import SO101CubeLiftTask

    a = goals["arrays"]
    if int(a["seeds"][slot]) != seed:
        raise cl.GuardError("G-goals: the slot's goal belongs to another reset")
    robot = make_robot(render)
    task = SO101CubeLiftTask(robot)
    frames = []

    def on_command(n, result):
        if render:
            frames.append(np.stack([robot.sim.render(c) for c in cl.CAMERAS]))

    rec = Recorder(robot, on_command)
    try:
        record = follow_lift(rec, task, seed)
    finally:
        robot.close()
    applied = np.asarray(rec.applied, np.float32).reshape(-1, cl.JOINTS)
    demo = a["demo_actions"][slot, : int(a["demo_lengths"][slot])]
    report = json.loads(json.dumps(goals["report"]["demonstrations"][str(seed)]))
    same = bool(
        np.array_equal(applied, demo)
        and record["success"] == report["success"]
        and (record.get("score") or {}).get("success_step") == report["success_step"]
    )
    arrays = {"applied": applied}
    if frames:
        arrays["frames"] = np.asarray(frames, np.uint8)
    np.savez_compressed(path, **arrays) if frames else np.savez(path, **arrays)
    if not same:
        raise cl.GuardError(f"G-F: F diverged from the goal demonstration on {seed}")
    score = record.get("score") or {}
    return {
        "arm": "F",
        "seed": int(seed),
        "slot": int(slot),
        "config": None,
        "success": bool(record["success"]),
        "termination": record["termination_reason"],
        "success_step": score.get("success_step"),
        "max_lift_m": score.get("max_lift_m"),
        "longest_hold_steps": score.get("longest_hold_steps"),
        "commands": len(applied),
        "stopped": "stopped" in rec.statuses,
        "clipped_commands": rec.statuses.count("clipped"),
        "equals_demonstration": same,
        "episode_seconds": time.perf_counter() - started,
        "file": path.name,
        "file_sha256": cl.sha256_file(path),
        "frames_kept": bool(frames),
        "pid": os.getpid(),
    }


# ----- the worker pool (§13) --
_CTX: dict = {}


def goal_context(goals: dict, featuriser: LiveFeaturiser | None) -> dict:
    a = goals["arrays"]
    n = len(a["seeds"])
    ctx = {"goals": goals, "derangement": cl.derangement(n) if n >= 2 else np.zeros(1, np.int64)}
    if featuriser is not None:
        raw = a["latents_raw"].reshape(n * cl.SUBGOALS, cl.TOKENS, cl.WIDTH)
        ctx["goal_latents"] = featuriser.standardise(raw).reshape(n, cl.SUBGOALS, cl.LATENT_DIM)
    return ctx


def init_worker(goals_dir: str, goals_sha256: str, device: str, threads: int = 2) -> None:
    """Each worker: its own CUDA context, deterministic setup, featuriser, models (lazily)."""
    import torch

    from embodied_jepa import devices

    torch.set_num_threads(int(threads))
    goals = cl.load_goal_store(goals_dir, goals_sha256)
    featuriser = None
    if device != "none":
        devices.configure_determinism(device, strict=True)
        featuriser = LiveFeaturiser(device)
    _CTX.clear()
    _CTX.update(goal_context(goals, featuriser))
    _CTX["featuriser"] = featuriser
    models: dict = {}

    def adapter(name, cost):
        if name not in models:
            models[name] = load_model(name, device)[0]
        return LatentCost(models[name], cost, action_blind=name == cl.TWIN)

    _CTX["adapter"] = adapter


def _worker_episode(task_spec: dict) -> dict:
    return run_episode(task_spec, _CTX)


def run_pool(
    specs: list[dict], *, goals_dir, goals_sha256, device: str, workers: int, on_result=None
) -> list[dict]:
    """Episodes in a spawn pool, **one fresh worker process per episode** (``maxtasksperchild=1``)
    for every pool size, 1 included: MuJoCo's EGL renderer is not bit-stable across a process's
    history (a frame rendered after earlier episodes in the same process can differ by one
    intensity level in a pixel; Stage 0 record, §7.2.4), so every episode starts from the same
    process state and depends only on its own spec, whatever the pool size or order."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        max(1, int(workers)),
        initializer=init_worker,
        initargs=(str(goals_dir), goals_sha256, device),
        maxtasksperchild=1,
    ) as pool:
        out = []
        for r in pool.imap(_worker_episode, specs, chunksize=1):
            out.append(r)
            if on_result:
                on_result(r)
        return out
