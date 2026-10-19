"""TASK-102 Δa's runtime (``docs/experiments/so101_approach_diagnostic_v1.md`` §2-§5): the goal
store of cohort L, the simulation copy that executes candidates, the oracle cost adapters, Λ0's
re-reading of Stage 0's A logs, Λ1's per-seed landscape and Λ2's approach episodes, and the
worker pool.

TASK-101's live featuriser, model loader, ``LatentCost``, projector, goal demonstration and
snapshot proxy, and TASK-102's ``HandoverRule``, are imported unchanged. NumPy at import; torch,
MuJoCo and the TASK-100 model are imported inside the functions that need them. Nothing in
``import embodied_jepa`` loads this module.

The oracle adapters (O-img, O-pose) read the episode simulation's state to execute candidates in
a copy. They are privileged diagnostic instruments, never controllers (§4, §8).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from embodied_jepa import so101_approach_diag as ad
from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_closed_loop_runtime as rt
from embodied_jepa import so101_hybrid as hy
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
OUT_NAME = "task102-approach-diag"


def header(kind: str, **extra) -> dict:
    report = rt.header(kind, **extra)
    report["task"], report["protocol"], report["study"] = ad.TASK, ad.PROTOCOL, ad.STUDY
    return report


# ----- the goal store of cohort L (§1) --
def build_goal_store(cohort: str, device: str, out, *, limit: int | None = None) -> dict:
    """TASK-101's goal demonstration on the cohort's seeds (keep rule with reserve), subgoal
    frames encoded with the live featuriser; written as a TASK-101-format goal store."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    report = header("goals", cohort=cohort, task_name=hy.LIFT_TASK_NAME)
    rt.check_frozen_task100(report)
    featuriser = rt.LiveFeaturiser(device)
    report["fit_report_sha256"] = featuriser.fit_report_sha256
    report["encoder_digest"] = featuriser.featuriser.digest
    demos: dict = {}
    c = ad.cohort(cohort)
    if limit is not None:
        if cohort != "debug":
            raise ad.GuardError("--limit is for the debug cohort only")
        ad.COHORTS["debug"] = ad.Cohort("debug", c.first, int(limit), None)

    def demonstrate(seed):
        ad.check_seed(cohort, seed)
        d = rt.demonstrate(seed, render=True)
        demos[seed] = d
        log(f"demo {seed}: {d['termination_reason']} step {d['success_step']} kept={d['kept']}")
        return d["kept"], d["reason"]

    kept = ad.keep_cohort(cohort, demonstrate)
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
            raise ad.GuardError(f"G-goals: {s}'s captures are not the four subgoals")
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
    keys = (
        "success",
        "termination_reason",
        "success_step",
        "stopped",
        "commands",
        "kept",
        "reason",
    )
    report["demonstrations"] = {str(s): {k: d[k] for k in keys} for s, d in demos.items()}
    report["goals_seconds"] = time.monotonic() - t0
    sha = cl.save_goal_store(out, arrays, report)
    log(f"goal store {sha[:12]}: kept {n}, rejected {len(kept['rejected'])}")
    return {"goals_sha256": sha, "kept": seeds, "rejected": kept["rejected"]}


# ----- the simulation copy (§3.3, §4, §5 G-rollout) --
class Scratch:
    """A copy of an episode's simulation: the same ``MjModel``, its own ``MjData`` and renderer.
    ``command`` repeats ``SO101Simulation.send_joint_targets``' arithmetic (targets interpolated
    linearly over the substeps from the previous targets), so from the same state it reaches the
    same joint positions as the real transport bit for bit (G-rollout; a test checks it)."""

    def __init__(self, sim, *, render: bool):
        mj = sim.mj
        self.mj, self.model = mj, sim.model
        self.data = mj.MjData(sim.model)
        self.substeps = int(sim.substeps)
        self.qadr, self.vadr = sim.qadr, sim.vadr
        self.spec = mj.mjtState.mjSTATE_INTEGRATION
        self.size = mj.mj_stateSize(sim.model, self.spec)
        self.renderer = (
            mj.Renderer(sim.model, height=sim.height, width=sim.width) if render else None
        )
        self.targets = np.zeros(sim.model.nu)

    def snapshot(self, sim) -> tuple[np.ndarray, np.ndarray]:
        state = np.empty(self.size)
        self.mj.mj_getState(self.model, sim.data, state, self.spec)
        return state, sim.targets.copy()

    def restore(self, snap) -> None:
        state, targets = snap
        self.mj.mj_setState(self.model, self.data, state, self.spec)
        self.targets = np.asarray(targets, np.float64).copy()

    def command(self, targets) -> None:
        targets = np.asarray(targets, np.float64)
        old = self.targets.copy()
        self.targets = targets.copy()
        for k in range(self.substeps):
            self.data.ctrl[:] = old + (targets - old) * ((k + 1) / self.substeps)
            self.mj.mj_step(self.model, self.data)

    def render(self, camera: str) -> np.ndarray:
        from embodied_jepa.so101_simulation import CAMERA_SOURCES

        self.renderer.update_scene(self.data, camera=CAMERA_SOURCES[camera])
        return self.renderer.render().copy()

    def frames(self) -> np.ndarray:
        return np.stack([self.render(c) for c in cl.CAMERAS])

    def run(self, snap, target_seq, *, frames: str = "none", velocity_stop=None) -> dict:
        """Restore ``snap`` and send each row of ``target_seq`` [T, 6]. ``frames``: ``none``,
        ``last`` or ``all`` (both cameras after the command). ``over_velocity`` records whether a
        command would have been refused by ``SO101Embodiment.execute``'s measured-velocity stop
        (checked before each command, as ``execute`` does); the copy executes it anyway."""
        self.restore(snap)
        seq = np.asarray(target_seq, np.float64)
        q = np.empty((len(seq), cl.JOINTS))
        over = False
        shots = []
        for t, row in enumerate(seq):
            if velocity_stop is not None and np.any(
                np.abs(self.data.qvel[self.vadr]) > velocity_stop
            ):
                over = True
            self.command(row)
            q[t] = self.data.qpos[self.qadr]
            if frames == "all" or (frames == "last" and t == len(seq) - 1):
                shots.append(self.frames())
        out = {"q": q, "over_velocity": over}
        if shots:
            out["frames"] = np.asarray(shots, np.uint8)
        return out

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class RecordingProjector(rt.Projector):
    """TASK-101's projector that also keeps the last call's per-step joint targets, so an oracle
    adapter executes exactly what ``execute`` would send. Its applied actions equal
    ``rt.Projector``'s bit for bit (``ad.project_chain``; a test checks it)."""

    def __call__(self, requested):
        applied, chain = ad.project_chain(
            np.asarray(requested, np.float32), self.targets, self.params
        )
        self.last_applied, self.last_targets = applied, chain
        return rt.SchemaCandidateProjection(applied, np.ones(applied.shape[:2], dtype=bool))


class OracleCost:
    """O-img / O-pose (§4): the planner-model adapter that executes every projected candidate in
    the simulation copy from the bound state. ``img``: the final frames rendered and encoded,
    latent error to subgoal 1 (float64, then float32 as ``LatentCost``); ``pose``:
    ``ad.pose_cost`` of the final joints. Only the last step is read by the planner (``last``
    form); every step carries the last step's cost."""

    def __init__(self, kind: str, scratch: Scratch, *, goal_q, goal_latent=None, featuriser=None,
                 velocity_stop=None):  # fmt: skip
        if kind not in ("img", "pose"):
            raise ContractError("oracle kind must be img or pose")
        if kind == "img" and (featuriser is None or goal_latent is None):
            raise ContractError("O-img needs the featuriser and subgoal 1's latent")
        self.kind, self.scratch, self.featuriser = kind, scratch, featuriser
        self.goal_q = np.asarray(goal_q, np.float64)
        self.goal_latent = None if goal_latent is None else np.asarray(goal_latent, np.float32)
        self.velocity_stop = velocity_stop
        self.snap = self.projector = None
        self.rollouts = 0
        self.over_velocity = 0

    def bind(self, snap, projector: RecordingProjector) -> None:
        self.snap, self.projector = snap, projector

    def predict(self, latent, candidates):
        c = np.asarray(candidates)
        if c.ndim != 4 or c.shape[0] != 1 or c.shape[-1] != cl.JOINTS:
            raise ContractError("candidates must be [1, K, H, 6]")
        if not np.array_equal(c, self.projector.last_applied):
            raise ad.GuardError("G-rollout: the scored candidates are not the projector's")
        chain = self.projector.last_targets[0]
        render = "last" if self.kind == "img" else "none"
        finals, shots = [], []
        for k in range(c.shape[1]):
            r = self.scratch.run(
                self.snap, chain[k], frames=render, velocity_stop=self.velocity_stop
            )
            finals.append(r["q"][-1])
            self.over_velocity += int(r["over_velocity"])
            if render == "last":
                shots.append(r["frames"][-1])
        self.rollouts += c.shape[1]
        out = {"q_end": np.asarray(finals), "h": c.shape[2]}
        if shots:
            s = np.asarray(shots)
            _, z = self.featuriser.encode({cam: s[:, j] for j, cam in enumerate(cl.CAMERAS)})
            out["z"] = z
        return out

    def distance(self, predictions, goal) -> np.ndarray:
        if self.kind == "img":
            cost = ad.part_errors(predictions["z"], self.goal_latent)[:, 0]
        else:
            cost = ad.pose_cost(predictions["q_end"], self.goal_q)
        if not np.isfinite(cost).all():
            raise ad.GuardError("G-finite: a non-finite oracle cost")
        out = np.repeat(cost.astype(np.float32)[:, None], predictions["h"], 1)
        return np.ascontiguousarray(out)[None]


# ----- Λ0: Stage 0's A logs (§2; CPU, reported only) --
def lambda0(stage0_dir) -> dict:
    """The true-frame cost along A's own approach trajectories, per configuration."""
    s0 = Path(stage0_dir)
    goals = cl.load_goal_store(s0 / "A-goals")
    a = goals["arrays"]
    moments = rt.load_moments()
    n = len(a["seeds"])
    g = rt.standardise(moments, a["latents_raw"][:, hy.SUBGOAL_1]).reshape(n, cl.LATENT_DIM)
    slot_of = {int(s): i for i, s in enumerate(a["seeds"])}
    out = {"goals_sha256": goals["sha256"], "configs": {}}
    for cfg in cl.CONFIGS:
        folder = s0 / "A-episodes" / f"H-approach@{cfg}"
        rows = []
        for path in sorted(folder.glob("*.npz")):
            seed = int(path.stem)
            slot = slot_of[seed]
            with np.load(path) as d:
                q, lat, cost = d["q"], d["latents_raw16"], d["cost"]
            z = rt.standardise(moments, lat)
            e = ad.part_errors(z, g[slot])
            rms = ad.arm_rms(q[: len(z)], a["joints"][slot, hy.SUBGOAL_1])
            i_close = int(np.argmin(rms))
            ratio = cost[: len(z)] / e[:, 0]
            rows.append(
                {
                    "seed": seed,
                    "steps": len(z),
                    "rho_full": ad.spearman(e[:, 0], rms),
                    "rho_wrist": ad.spearman(e[:, 1], rms),
                    "rho_overview": ad.spearman(e[:, 2], rms),
                    "c_true_reset": e[0].tolist(),
                    "c_true_closest": e[i_close].tolist(),
                    "c_true_last": e[-1].tolist(),
                    "rms_reset": float(rms[0]),
                    "rms_closest": float(rms[i_close]),
                    "rms_last": float(rms[-1]),
                    "planned_over_true_median": float(np.median(ratio)),
                }
            )
        med = {
            k: ad.median(r[k] for r in rows)
            for k in (
                "rho_full",
                "rho_wrist",
                "rho_overview",
                "rms_reset",
                "rms_closest",
                "rms_last",
                "planned_over_true_median",
            )  # fmt: skip
        }
        for which in ("reset", "closest", "last"):
            for p, part in enumerate(ad.PARTS):
                med[f"c_true_{which}_{part}"] = ad.median(r[f"c_true_{which}"][p] for r in rows)
        out["configs"][cfg] = {"episodes": len(rows), "median": med, "episodes_detail": rows}
    return out


# ----- workers (§3, §4) --
_CTX: dict = {}


def init_worker(goals_dir: str, goals_sha256: str, device: str, threads: int = 2) -> None:
    import torch

    from embodied_jepa import devices

    torch.set_num_threads(int(threads))
    goals = cl.load_goal_store(goals_dir, goals_sha256)
    featuriser = None
    if device != "none":
        devices.configure_determinism(device, strict=True)
        featuriser = rt.LiveFeaturiser(device)
    _CTX.clear()
    _CTX.update(rt.goal_context(goals, featuriser))
    _CTX["featuriser"] = featuriser
    _CTX["device"] = device
    models: dict = {}

    def model(name):
        if name not in models:
            models[name] = rt.load_model(name, device)[0]
        return models[name]

    _CTX["model"] = model


def _velocity_stop(robot) -> float:
    return float(robot.manifest["measured_joint_velocity_stop_rad_s"])


# ----- Λ1: the landscape on one seed (§3) --
def _place(sim, q) -> None:
    """Set the six joints to ``q`` (clipped to the transport's limits), zero velocities, targets
    and controls to ``q``, forward; then ``SETTLE_COMMANDS`` hold commands."""
    q = np.clip(np.asarray(q, np.float64), sim.limits[:, 0], sim.limits[:, 1])
    sim.data.qpos[sim.qadr] = q
    sim.data.qvel[sim.vadr] = 0.0
    sim.targets[:] = q
    sim.data.ctrl[:] = q
    sim.mj.mj_forward(sim.model, sim.data)
    for _ in range(ad.SETTLE_COMMANDS):
        sim.send_joint_targets(q, joint_names=sim.joint_names, deadline=float(sim.data.time) + 1.0)


def _hold_costs(adapter, z, targets, params, goal) -> np.ndarray:
    """C_J-hold: J's predicted part errors [16, 3] of 16 hold commands from latent ``z``."""
    hold = ad.hold_sequence(targets, params, ad.CANDIDATE_H)
    applied, _ = cl.project_sequences(hold[None], targets, params)
    pred = adapter.predict(z, applied[None])["pred"].double().cpu().numpy()[0]
    return ad.part_errors(pred, goal)


def landscape_seed(seed: int, slot: int, ctx: dict, out_dir) -> dict:
    """Λ1 on one kept L seed: demonstration-path states, line states, and 64 executed
    candidates at the five candidate states. Writes ``<out_dir>/<seed>.npz``."""
    from embodied_jepa.so101_followers import follow_lift

    out = Path(out_dir) / f"{seed}.npz"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    started = time.perf_counter()
    a = ctx["goals"]["arrays"]
    if int(a["seeds"][slot]) != seed:
        raise ad.GuardError("G-goals: the slot's goal belongs to another reset")
    q1 = np.asarray(a["joints"][slot, hy.SUBGOAL_1], np.float64)
    g1 = ctx["goal_latents"][slot, hy.SUBGOAL_1]
    featuriser = ctx["featuriser"]
    adapter = rt.LatentCost(ctx["model"](cl.PRIMARY), "last")
    # the demonstration path
    robot = rt.make_robot(True)
    task = rt.lift_task(robot)
    snap = rt._Snapshots(robot, ad.DEMO_STATES, render=True)
    try:
        follow_lift(snap, task, int(seed))
    finally:
        robot.close()
    applied = np.asarray(snap.applied, np.float32).reshape(-1, cl.JOINTS)
    demo = a["demo_actions"][slot, : int(a["demo_lengths"][slot])]
    if not np.array_equal(applied, demo):
        raise ad.GuardError(f"G-replay: the replay of {seed} differs from its goal demonstration")
    params = None
    demo_frames = np.stack(
        [np.stack([snap.states[t]["frames"][c] for c in cl.CAMERAS]) for t in ad.DEMO_STATES]
    )
    demo_q = np.stack([snap.states[t]["q"] for t in ad.DEMO_STATES])
    demo_targets = np.stack([snap.states[t]["targets"] for t in ad.DEMO_STATES])
    # the line states and the candidates
    robot = rt.make_robot(True)
    task = rt.lift_task(robot)
    sim = robot.sim
    scratch = Scratch(sim, render=True)
    vstop = _velocity_stop(robot)
    line, cand = [], []
    try:
        task.reset(int(seed))
        params = cl.PreviewParams.of(robot)
        q0 = sim.data.qpos[sim.qadr].copy()
        d0 = float(ad.arm_rms(q0, q1))
        for index, d in enumerate(ad.line_distances(d0)):
            task.reset(int(seed))
            cube0 = sim.data.body("cube").xpos.copy()
            _place(sim, ad.line_point(q0, q1, d, d0))
            moved = float(np.linalg.norm(sim.data.body("cube").xpos - cube0))
            q = sim.data.qpos[sim.qadr].copy()
            row = {
                "nominal": -1.0 if d == ad.RESET else float(d),
                "q": q,
                "targets": sim.targets.copy(),
                "d_meas": float(ad.arm_rms(q, q1)),
                "jaw_err": float(q[cl.JAW] - q1[cl.JAW]),
                "cube_moved_m": moved,
                "frames": np.stack([sim.render(c) for c in cl.CAMERAS]),
            }
            line.append(row)
            if d not in ad.CANDIDATE_STATES or moved > ad.CUBE_MOVED_M:
                continue
            state = scratch.snapshot(sim)
            requested = ad.candidate_set(seed, index, sim.targets, q1, params)
            app, chain = ad.project_chain(requested, sim.targets, params)
            shots = np.empty(
                (ad.CANDIDATES, ad.CANDIDATE_H, 2, cl.IMAGE_SIZE, cl.IMAGE_SIZE, 3), np.uint8
            )
            q_end = np.empty((ad.CANDIDATES, ad.CANDIDATE_H, cl.JOINTS))
            over = np.zeros(ad.CANDIDATES, bool)
            for k in range(ad.CANDIDATES):
                r = scratch.run(state, chain[k], frames="all", velocity_stop=vstop)
                shots[k], q_end[k], over[k] = r["frames"], r["q"], r["over_velocity"]
            flat = shots.reshape(-1, 2, cl.IMAGE_SIZE, cl.IMAGE_SIZE, 3)
            _, z_exec = featuriser.encode({c: flat[:, j] for j, c in enumerate(cl.CAMERAS)})
            true_err = ad.part_errors(z_exec, g1).reshape(ad.CANDIDATES, ad.CANDIDATE_H, 3)
            _, z_state = featuriser.encode({c: row["frames"][j][None] for j, c in
                                            enumerate(cl.CAMERAS)})  # fmt: skip
            pred = adapter.predict(z_state[0], app[None])["pred"].double().cpu().numpy()
            j_err = ad.part_errors(pred, g1)
            # G-rollout: the approach candidate through the real transport equals the copy's
            real_q, stopped = [], False
            for step in requested[0]:
                robot.observe()
                res = robot.execute(step)
                if res.status == "stopped":
                    stopped = True
                    break
                real_q.append(sim.data.qpos[sim.qadr].copy())
            real_q = np.asarray(real_q)
            if not np.array_equal(real_q, q_end[0][: len(real_q)]):
                raise ad.GuardError(f"G-rollout: the copy differs from the transport on {seed}")
            cand.append(
                {
                    "nominal": row["nominal"],
                    "requested": requested,
                    "applied": app,
                    "q_end": q_end,
                    "d_end": ad.arm_rms(q_end, q1),
                    "jaw_end": q_end[..., cl.JAW] - q1[cl.JAW],
                    "true_err": true_err,
                    "j_err": j_err,
                    "over_velocity": over,
                    "rollout_check_commands": len(real_q),
                    "rollout_check_stopped": stopped,
                }
            )
    finally:
        scratch.close()
        robot.close()
    line_frames = np.stack([r["frames"] for r in line])
    all_frames = np.concatenate([demo_frames, line_frames])
    _, z_all = featuriser.encode({c: all_frames[:, j] for j, c in enumerate(cl.CAMERAS)})
    c_true = ad.part_errors(z_all, g1)
    targets_all = np.concatenate([demo_targets, np.stack([r["targets"] for r in line])])
    c_hold = np.stack(
        [_hold_costs(adapter, z_all[i], targets_all[i], params, g1) for i in range(len(z_all))]
    )
    nd = len(ad.DEMO_STATES)
    arrays = {
        "demo_t": np.asarray(ad.DEMO_STATES),
        "demo_q": demo_q,
        "demo_d": ad.arm_rms(demo_q, q1),
        "demo_jaw_err": demo_q[:, cl.JAW] - q1[cl.JAW],
        "demo_c_true": c_true[:nd],
        "demo_c_hold": c_hold[:nd],
        "demo_frames": demo_frames,
        "line_nominal": np.asarray([r["nominal"] for r in line]),
        "line_q": np.stack([r["q"] for r in line]),
        "line_d": np.asarray([r["d_meas"] for r in line]),
        "line_jaw_err": np.asarray([r["jaw_err"] for r in line]),
        "line_cube_moved_m": np.asarray([r["cube_moved_m"] for r in line]),
        "line_c_true": c_true[nd:],
        "line_c_hold": c_hold[nd:],
        "line_frames": line_frames,
        "d0": np.asarray(d0),
        "q1": q1,
    }
    for key in ("nominal", "requested", "applied", "d_end", "jaw_end", "true_err", "j_err",
                "over_velocity", "rollout_check_commands", "rollout_check_stopped"):  # fmt: skip
        arrays[f"cand_{key}"] = np.asarray([c[key] for c in cand])
    np.savez_compressed(out, **arrays)
    return {
        "seed": int(seed),
        "slot": int(slot),
        "d0": d0,
        "line_states": len(line),
        "excluded": int(sum(r["cube_moved_m"] > ad.CUBE_MOVED_M for r in line)),
        "candidate_states": len(cand),
        "over_velocity_candidates": int(sum(int(c["over_velocity"].sum()) for c in cand)),
        "rollout_checked_commands": int(sum(c["rollout_check_commands"] for c in cand)),
        "seconds": time.perf_counter() - started,
        "file": out.name,
        "file_sha256": cl.sha256_file(out),
        "pid": os.getpid(),
    }


# ----- Λ2: one approach episode (§4) --
def _planner(arm: str, seed: int):
    from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
    from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner

    cfg = ad.check_arm(arm)
    return SchemaCEMPlanner(
        SchemaCEMConfig(
            action_schema=JOINT_DELTA_SO101_V0,
            horizon=int(cfg["horizon"]),
            samples=int(cfg["samples"]),
            iterations=int(cfg["iterations"]),
            elites=int(cfg["elites"]),
            seed=ad.planner_seed(seed),
            action_penalty=cl.CEM["action_penalty"],
            minimum_std=cl.CEM["minimum_std"],
            project_candidates=True,
        )
    )


def approach_episode(spec: dict, ctx: dict) -> dict:
    """Phase 1 alone with ``spec['arm']``'s cost and budget toward subgoal 1, until the hand-over
    (τ_h = 0.08 rad) or the 60-command timeout. Writes ``<output>/<seed>.npz``."""
    import torch

    arm, seed, slot = spec["arm"], int(spec["seed"]), int(spec["slot"])
    cfg = ad.check_arm(arm)
    path = Path(spec["output"]) / f"{seed}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    a = ctx["goals"]["arrays"]
    if int(a["seeds"][slot]) != seed:
        raise ad.GuardError("G-goals: the slot's goal belongs to another reset")
    started = time.perf_counter()
    q1 = np.asarray(a["joints"][slot, hy.SUBGOAL_1], np.float64)
    render = cfg["cost"] != "pose"
    featuriser = ctx["featuriser"]
    g1 = ctx["goal_latents"][slot, hy.SUBGOAL_1] if render else np.zeros(cl.LATENT_DIM, np.float32)
    robot = rt.make_robot(render)
    task = rt.lift_task(robot)
    task.reset(seed)
    sim = robot.sim
    params = cl.PreviewParams.of(robot)
    sentinel = rt.TruthSentinel(sim)
    timeout = int(spec.get("timeout", hy.T1))
    if timeout != hy.T1 and not ad.cohort("debug").contains(seed):
        raise ad.GuardError("a shortened timeout is for debug seeds only")
    rule = hy.HandoverRule(q1, ad.TAU_H, timeout=timeout)
    planner = _planner(arm, seed)
    scratch = oracle = None
    if cfg["cost"] == "J":
        adapter = rt.LatentCost(ctx["model"](cl.PRIMARY), "last")
    else:
        scratch = Scratch(sim, render=cfg["cost"] == "img")
        oracle = adapter = OracleCost(
            cfg["cost"],
            scratch,
            goal_q=q1,
            goal_latent=g1 if cfg["cost"] == "img" else None,
            featuriser=featuriser,
            velocity_stop=_velocity_stop(robot),
        )
    rows: dict[str, list] = {
        k: []
        for k in (
            "q",
            "qvel",
            "targets",
            "requested",
            "applied",
            "projected",
            "status",
            "cost",
            "plan_seconds",
            "c_true",
        )  # fmt: skip
    }
    latents = []
    termination = "running"
    rollout_checks = 0
    try:
        for _ in range(hy.T1 + 2):
            obs = robot.observe()
            imgs = {c: obs.images[c][0] for c in cl.CAMERAS}
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
            if cfg["cost"] == "J":
                projector = rt.Projector(params, targets)
            else:
                projector = RecordingProjector(params, targets)
                pre = scratch.snapshot(sim)
                oracle.bind(pre, projector)
            t0 = time.perf_counter()
            with sentinel.controller():
                plan = planner.plan(adapter, z, g1, projector=projector)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            if not np.isfinite(plan.costs).all():
                raise ad.GuardError("G-finite: a non-finite planned cost")
            requested = np.asarray(plan.requested_actions[0, 0], np.float32).copy()
            projected = np.asarray(plan.actions[0, 0], np.float32).copy()
            result = robot.execute(requested)
            if result.status != "stopped" and not np.array_equal(result.applied_action, projected):
                raise ad.GuardError("G-projection: the executed action differs from the scored one")
            if oracle is not None and result.status != "stopped":
                scratch.restore(pre)
                scratch.command(sim.targets)
                if not (
                    np.array_equal(scratch.data.qpos, sim.data.qpos)
                    and np.array_equal(scratch.data.qvel, sim.data.qvel)
                ):
                    raise ad.GuardError(f"G-rollout: the copy differs from the transport on {seed}")
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
            rows["cost"].append(float(plan.costs[0]))
            rows["plan_seconds"].append(seconds)
            rows["c_true"].append(ad.part_errors(z, g1) if render else np.full(3, np.nan))
            if result.status == "stopped":
                termination = f"stopped: {result.reason}"
                break
        else:  # pragma: no cover -- the rule's timeout ends every episode first
            raise ad.GuardError("G-steps: the approach outran its command budget")
    finally:
        if scratch is not None:
            scratch.close()
        robot.close()
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    if latents:
        arrays["latents_raw16"] = np.asarray(latents, np.float16)
    np.savez(path, **arrays)
    event = rule.event or {}
    rms = ad.arm_rms(arrays["q"], q1) if len(arrays["q"]) else np.array([np.nan])
    i = int(np.nanargmin(rms))
    return {
        "arm": arm,
        "seed": seed,
        "slot": slot,
        "termination": termination,
        "handover": event.get("how"),
        "by_tolerance": event.get("how") == "tolerance",
        "phase1_commands": event.get("phase1_commands"),
        "handover_arm_rms": event.get("arm_rms_rad"),
        "handover_jaw_error": event.get("jaw_error_rad"),
        "closest_arm_rms": float(rule.closest),
        "jaw_error_at_closest": float(arrays["q"][i, cl.JAW] - q1[cl.JAW]) if len(rms) else None,
        "commands": len(arrays["status"]),
        "plan_seconds_median": float(np.median(arrays["plan_seconds"])),
        "oracle_rollouts": oracle.rollouts if oracle else 0,
        "oracle_over_velocity": oracle.over_velocity if oracle else 0,
        "rollout_checks": rollout_checks,
        "episode_seconds": time.perf_counter() - started,
        "file": path.name,
        "file_sha256": cl.sha256_file(path),
        "pid": os.getpid(),
    }


def _worker(job: dict) -> dict:
    if job["kind"] == "landscape":
        return landscape_seed(int(job["seed"]), int(job["slot"]), _CTX, job["output"])
    return approach_episode(job, _CTX)


def run_pool(jobs, *, goals_dir, goals_sha256, device: str, workers: int, on_result=None):
    """One fresh spawn process per job (TASK-101's EGL precedent)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        max(1, int(workers)),
        initializer=init_worker,
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
