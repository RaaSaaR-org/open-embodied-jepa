"""TASK-102 A1's runtime (``docs/experiments/so101_hybrid_lift_v1.md`` §18.3): cohort Q's goal
store, the per-step pose oracle O-pose, one phase-1 episode of a grid configuration with the
scripted jaw and the arm-only hand-over, and the worker pool.

TASK-101's live featuriser, model loader, ``LatentCost``, projector, goal demonstration and
truth sentinel, and Δa's simulation copy (``Scratch``), recording projector and worker
initialiser are imported unchanged. NumPy at import; torch, MuJoCo and the TASK-100 model are
imported inside the functions that need them. Nothing in ``import embodied_jepa`` loads this
module.

O-pose reads the episode simulation's state to execute candidates in a copy. It is a privileged
planner check, never a controller (§18.3).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from embodied_jepa import so101_approach_diag_runtime as ar
from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_closed_loop_runtime as rt
from embodied_jepa import so101_hybrid as hy
from embodied_jepa import so101_hybrid_a1 as a1
from embodied_jepa.contracts import ContractError

ROOT = rt.ROOT
shared_root, utc, log, git, tree_dirty, write_json = (
    rt.shared_root,
    rt.utc,
    rt.log,
    rt.git,
    rt.tree_dirty,
    rt.write_json,
)
OUT_NAME = "task102-a1"


def header(kind: str, **extra) -> dict:
    report = rt.header(kind, **extra)
    report["task"], report["protocol"], report["section"] = a1.TASK, a1.PROTOCOL, a1.SECTION
    report["study"] = a1.STUDY
    return report


def check_pins(report: dict, *, counted: bool) -> None:
    """TASK-100's, TASK-101's and TASK-102's pins (``hr.check_pins``), Δa's files and
    ``schema_planning.py`` (§18.6); a counted job also needs a clean tracked tree."""
    from embodied_jepa import so101_hybrid_runtime as hr

    hr.check_pins(report)
    bad = a1.differing_pins(ROOT)
    report["a1_pins_ok"] = not bad
    if bad:
        raise a1.GuardError(f"G-frozen: pinned files differ: {bad}")
    dirty = tree_dirty()
    report["tracked_tree_dirty"] = dirty
    if counted and dirty:
        raise a1.GuardError("G-tree: a counted job needs a clean tracked tree")


# ----- cohort Q's goal store (§18.3) --
def build_goal_store(cohort: str, device: str, out, *, limit: int | None = None) -> dict:
    """TASK-101's goal demonstration on the cohort's seeds (keep rule with reserve), subgoal
    frames encoded with the live featuriser; written as a TASK-101-format goal store (Δa's
    ``build_goal_store`` with A1's cohorts)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    report = header("goals", cohort=cohort, task_name=hy.LIFT_TASK_NAME)
    rt.check_frozen_task100(report)
    featuriser = rt.LiveFeaturiser(device)
    report["fit_report_sha256"] = featuriser.fit_report_sha256
    report["encoder_digest"] = featuriser.featuriser.digest
    demos: dict = {}
    c = a1.cohort(cohort)
    if limit is not None:
        if cohort != "debug":
            raise a1.GuardError("--limit is for the debug cohort only")
        a1.COHORTS["debug"] = a1.Cohort("debug", c.first, int(limit), None)

    def demonstrate(seed):
        a1.check_seed(cohort, seed)
        d = rt.demonstrate(seed, render=True)
        demos[seed] = d
        log(f"demo {seed}: {d['termination_reason']} step {d['success_step']} kept={d['kept']}")
        return d["kept"], d["reason"]

    kept = a1.keep_cohort(cohort, demonstrate)
    seeds = kept["kept"]
    n = len(seeds)
    size = cl.IMAGE_SIZE
    frames = np.zeros((n, cl.SUBGOALS, 2, size, size, 3), np.uint8)
    raw = np.zeros((n, cl.SUBGOALS, cl.TOKENS, cl.WIDTH), np.float16)
    joints = np.zeros((n, cl.SUBGOALS, cl.JOINTS))
    cube = np.zeros((n, cl.SUBGOALS, 3))
    site = np.zeros((n, cl.SUBGOALS, 3))
    xmat = np.zeros((n, cl.SUBGOALS, 9))
    lengths = np.array([demos[s]["commands"] for s in seeds], np.int64)
    actions = np.full((n, int(lengths.max()), cl.JOINTS), np.nan, np.float32)
    for i, s in enumerate(seeds):
        caps = demos[s]["captured"]
        if [cap["command"] for cap in caps] != list(cl.SUBGOAL_COMMANDS):
            raise a1.GuardError(f"G-goals: {s}'s captures are not the four subgoals")
        for k, cap in enumerate(caps):
            frames[i, k] = cap["frames"]
            joints[i, k] = cap["joints"]
            cube[i, k] = cap["cube_cm"]
            site[i, k] = cap["site"]
            xmat[i, k] = cap["site_xmat"]
        raw[i] = featuriser.raw16({cam: frames[i, :, j] for j, cam in enumerate(cl.CAMERAS)})
        actions[i, : lengths[i]] = demos[s]["applied"]
    arrays = {
        "seeds": np.asarray(seeds, np.int64),
        "slots": np.arange(n, dtype=np.int64),
        "frames": frames,
        "latents_raw": raw,
        "joints": joints,
        "cube_cm": cube,
        "site_m": site,
        "site_xmat": xmat,
        "demo_actions": actions,
        "demo_lengths": lengths,
    }
    report |= {"kept": seeds, "rejected": kept["rejected"], "drawn": kept["drawn"]}
    keys = ("success", "termination_reason", "success_step", "stopped", "commands", "kept",
            "reason")  # fmt: skip
    report["demonstrations"] = {str(s): {k: d[k] for k in keys} for s, d in demos.items()}
    report["subgoal1_jaw_rad"] = [float(x) for x in joints[:, hy.SUBGOAL_1, cl.JAW]]
    report["goals_seconds"] = time.monotonic() - t0
    sha = cl.save_goal_store(out, arrays, report)
    log(f"goal store {sha[:12]}: kept {n}, rejected {len(kept['rejected'])}")
    return {"goals_sha256": sha, "kept": seeds, "rejected": kept["rejected"]}


# ----- O-pose, per step (§18.3) --
class PoseOracle:
    """The planner-model adapter that executes every projected candidate in the simulation copy
    from the bound state and returns ``a1.form_steps(a1.arm_pose_steps(q_t))`` for every command
    t of the horizon (no jaw term)."""

    def __init__(self, scratch: ar.Scratch, *, goal_q, form: str, velocity_stop=None):
        if form not in a1.FORMS:
            raise ContractError(f"form must be one of {a1.FORMS}")
        self.scratch, self.form = scratch, form
        self.goal_q = np.asarray(goal_q, np.float64)
        self.velocity_stop = velocity_stop
        self.snap = self.projector = None
        self.rollouts = 0
        self.over_velocity = 0

    def bind(self, snap, projector: ar.RecordingProjector) -> None:
        self.snap, self.projector = snap, projector

    def predict(self, latent, candidates):
        c = np.asarray(candidates)
        if c.ndim != 4 or c.shape[0] != 1 or c.shape[-1] != cl.JOINTS:
            raise ContractError("candidates must be [1, K, H, 6]")
        if not np.array_equal(c, self.projector.last_applied):
            raise a1.GuardError("G-rollout: the scored candidates are not the projector's")
        chain = self.projector.last_targets[0]
        qs = []
        for k in range(c.shape[1]):
            r = self.scratch.run(self.snap, chain[k], velocity_stop=self.velocity_stop)
            qs.append(r["q"])
            self.over_velocity += int(r["over_velocity"])
        self.rollouts += c.shape[1]
        return {"q": np.asarray(qs)}

    def distance(self, predictions, goal) -> np.ndarray:
        cost = a1.form_steps(a1.arm_pose_steps(predictions["q"], self.goal_q), self.form)
        if not np.isfinite(cost).all():
            raise a1.GuardError("G-finite: a non-finite oracle cost")
        return np.ascontiguousarray(cost.astype(np.float32))[None]


# ----- one phase-1 episode of a grid configuration (§18.2, §18.3) --
def episode(spec: dict, ctx: dict) -> dict:
    """Phase 1 alone with ``spec['config']`` and ``spec['cost']`` toward subgoal 1, with the
    scripted jaw and the arm-only hand-over (τ_h = 0.08 rad), until the hand-over or the
    60-command timeout. Writes ``<output>/<seed>.npz``."""
    import torch

    name, cost, seed, slot = spec["config"], spec["cost"], int(spec["seed"]), int(spec["slot"])
    cfg = a1.check_config(name)
    if cost not in a1.COSTS:
        raise ContractError(f"cost must be one of {a1.COSTS}")
    path = Path(spec["output"]) / f"{seed}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    a = ctx["goals"]["arrays"]
    if int(a["seeds"][slot]) != seed:
        raise a1.GuardError("G-goals: the slot's goal belongs to another reset")
    started = time.perf_counter()
    q1 = np.asarray(a["joints"][slot, hy.SUBGOAL_1], np.float64)
    render = cost == "J"
    featuriser = ctx["featuriser"]
    g1 = ctx["goal_latents"][slot, hy.SUBGOAL_1] if render else np.zeros(cl.LATENT_DIM, np.float32)
    robot = rt.make_robot(render)
    task = rt.lift_task(robot)
    task.reset(seed)
    sim = robot.sim
    params = cl.PreviewParams.of(robot)
    sentinel = rt.TruthSentinel(sim)
    timeout = int(spec.get("timeout", a1.T1))
    if timeout != a1.T1 and not a1.cohort("debug").contains(seed):
        raise a1.GuardError("a shortened timeout is for debug seeds only")
    rule = a1.ArmHandoverRule(q1, a1.TAU_H, timeout=timeout)
    pcfg = a1.planner_config(name, q1, seed)
    planner = a1.ArmCEMPlanner(pcfg)
    gripper = a1.scripted_gripper(q1)
    k_exec, warm = int(cfg["execute"]), bool(cfg["warm"])
    scratch = oracle = None
    if cost == "J":
        adapter = rt.LatentCost(ctx["model"](cl.PRIMARY), cfg["form"])
    else:
        scratch = ar.Scratch(sim, render=False)
        oracle = adapter = PoseOracle(
            scratch, goal_q=q1, form=cfg["form"], velocity_stop=ar._velocity_stop(robot)
        )
    keys = ("q", "qvel", "targets", "requested", "applied", "projected", "status", "replanned",
            "cost", "plan_seconds")  # fmt: skip
    rows: dict[str, list] = {k: [] for k in keys}
    latents = []
    termination = "running"
    rollout_checks = plans = 0
    queue: list = []
    previous = None
    executed_from_plan = 0
    try:
        for _ in range(a1.T1 + 2):
            obs = robot.observe()
            imgs = {c: obs.images[c][0] for c in cl.CAMERAS} if render else None
            q = np.asarray(obs.state.values[0, : cl.JOINTS], np.float64)
            targets = sim.targets.copy()
            z = np.zeros(cl.LATENT_DIM, np.float32)
            if render:
                raw16, zz = featuriser.encode({c: imgs[c][None] for c in cl.CAMERAS})
                latents.append(raw16[0])
                z = zz[0]
            if rule.update(q) == 2:
                termination = "handover"
                break
            replanned = not queue
            seconds, planned_cost = np.nan, np.nan
            if replanned:
                if cost == "J":
                    projector = rt.Projector(params, targets)
                else:
                    projector = ar.RecordingProjector(params, targets)
                    oracle.bind(scratch.snapshot(sim), projector)
                init = None
                if warm and previous is not None:
                    init = a1.shifted_mean(previous, executed_from_plan, pcfg)
                t0 = time.perf_counter()
                with sentinel.controller():
                    plan = planner.plan(adapter, z, g1, projector=projector, init_mean=init)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                plans += 1
                if not np.isfinite(plan.costs).all():
                    raise a1.GuardError("G-finite: a non-finite planned cost")
                planned_cost = float(plan.costs[0])
                req_seq = np.asarray(plan.requested_actions[0], np.float32)
                proj_seq = np.asarray(plan.actions[0], np.float32)
                if not np.all(req_seq[:, cl.JAW] == np.float32(gripper)):
                    raise a1.GuardError("G-jaw: a planned gripper command is not the scripted one")
                previous = req_seq.copy()
                executed_from_plan = 0
                queue = [(req_seq[i].copy(), proj_seq[i].copy()) for i in range(k_exec)]
            requested, projected = queue.pop(0)
            pre = scratch.snapshot(sim) if oracle is not None else None
            result = robot.execute(requested)
            executed_from_plan += 1
            if result.status != "stopped" and not np.array_equal(result.applied_action, projected):
                raise a1.GuardError("G-projection: the executed action differs from the scored one")
            if oracle is not None and result.status != "stopped":
                scratch.restore(pre)
                scratch.command(sim.targets)
                if not (
                    np.array_equal(scratch.data.qpos, sim.data.qpos)
                    and np.array_equal(scratch.data.qvel, sim.data.qvel)
                ):
                    raise a1.GuardError(f"G-rollout: the copy differs from the transport on {seed}")
                rollout_checks += 1
            rows["q"].append(q)
            rows["qvel"].append(np.asarray(obs.state.values[0, cl.JOINTS :], np.float64))
            rows["targets"].append(targets)
            rows["requested"].append(requested)
            rows["applied"].append(
                np.full(6, np.nan, np.float32)
                if result.applied_action is None
                else np.asarray(result.applied_action, np.float32)
            )
            rows["projected"].append(projected)
            rows["status"].append(rt.STATUS_CODES[result.status])
            rows["replanned"].append(replanned)
            rows["cost"].append(planned_cost)
            rows["plan_seconds"].append(seconds)
            if result.status == "stopped":
                termination = f"stopped: {result.reason}"
                break
        else:  # pragma: no cover -- the rule's timeout ends every episode first
            raise a1.GuardError("G-steps: the approach outran its command budget")
    finally:
        if scratch is not None:
            scratch.close()
        robot.close()
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays["q_last"] = np.asarray(q)
    if latents:
        arrays["latents_raw16"] = np.asarray(latents, np.float16)
    np.savez(path, **arrays)
    event = rule.event or {}
    q_all = np.concatenate((arrays["q"].reshape(-1, cl.JOINTS), np.asarray(q)[None]))
    rms = a1.ad.arm_rms(q_all, q1)
    i = int(np.argmin(rms))
    ps = arrays["plan_seconds"][np.isfinite(arrays["plan_seconds"])]
    return {
        "config": name,
        "cost": cost,
        "seed": seed,
        "slot": slot,
        "termination": termination,
        "handover": event.get("how"),
        "by_tolerance": event.get("how") == "tolerance",
        "phase1_commands": event.get("phase1_commands"),
        "handover_arm_rms": event.get("arm_rms_rad"),
        "handover_jaw_error": event.get("jaw_error_rad"),
        "closest_arm_rms": float(rule.closest),
        "jaw_error_at_closest": float(q_all[i, cl.JAW] - q1[cl.JAW]),
        "longest_near": int(rule.longest_near),
        "subgoal1_jaw": float(q1[cl.JAW]),
        "scripted_gripper": gripper,
        "commands": len(arrays["status"]),
        "plans": plans,
        "plan_seconds_median": float(np.median(ps)) if len(ps) else None,
        "seconds_per_command": (
            float(np.sum(ps) / len(arrays["status"])) if len(arrays["status"]) else None
        ),
        "oracle_rollouts": oracle.rollouts if oracle else 0,
        "oracle_over_velocity": oracle.over_velocity if oracle else 0,
        "rollout_checks": rollout_checks,
        "episode_seconds": time.perf_counter() - started,
        "file": path.name,
        "file_sha256": cl.sha256_file(path),
        "pid": os.getpid(),
    }


def _worker(job: dict) -> dict:
    return episode(job, ar._CTX)


def run_pool(jobs, *, goals_dir, goals_sha256, device: str, workers: int, on_result=None):
    """One fresh spawn process per job (TASK-101's EGL precedent); Δa's worker initialiser."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        max(1, int(workers)),
        initializer=ar.init_worker,
        initargs=(str(goals_dir), goals_sha256, device),
        maxtasksperchild=1,
    ) as pool:
        out = []
        for r in pool.imap(_worker, jobs, chunksize=1):
            out.append(r)
            if on_result:
                on_result(r)
        return out


def dump(path, payload) -> str:
    return write_json(path, json.loads(json.dumps(payload, default=cl._json_default)))
