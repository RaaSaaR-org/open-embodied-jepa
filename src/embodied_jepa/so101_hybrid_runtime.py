"""TASK-102's runtime (``docs/experiments/so101_hybrid_lift_v1.md`` §3-§7, §12): G_loc's training
windows, the hybrid controllers, the follower-from-hand-over path, the episode loop, the worker
pool and the G-zero guards.

Phase 1 is TASK-101's CEM over J (``so101_closed_loop_runtime``'s live featuriser, cost adapter
and projector, imported unchanged) aimed at subgoal 1 only; the hand-over is
``so101_hybrid.HandoverRule``; phase 2 is G_loc (``so101_grasp_policy``) or, for W+F, follower
v1's descent, close and lift from the hand-over pose (the envelope's "pre" path).

NumPy at import; torch, MuJoCo, the TASK-100 model and G_loc are imported inside the functions
that need them. Nothing in ``import embodied_jepa`` loads this module.

What each learned controller reads: the current ``wrist_rgb`` and ``overview_rgb`` frames (one
encoder pass per command serves both phases), its measured joint positions and velocities, its
own joint-position targets, the clock since the hand-over and, in phase 1, subgoal 1's latent and
joints. While a controller acts, TASK-101's sentinel makes every simulator-truth accessor raise.
The privileged arms (S+G's scripted approach, W+F's follower, F) read truth only through aims the
runner computes at the reset, outside the sentinel.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

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


def header(kind: str, **extra) -> dict:
    report = rt.header(kind, **extra)
    report["task"], report["protocol"] = hy.TASK, hy.PROTOCOL
    return report


def sha256_file(path) -> str:
    return cl.sha256_file(path)


def check_pins(report: dict, *, gated: bool = False) -> None:
    """§12: TASK-100's ``frozen_files_check``, TASK-101's pinned files byte-identical, and this
    task's manifest (FROZEN with every pin matching, for gated jobs)."""
    rt.check_frozen_task100(report)
    pins = json.loads((ROOT / cl.MANIFEST).read_text())
    files = pins["files_sha256"] | pins["composed_files_sha256"]
    bad = [rel for rel, pin in files.items() if sha256_file(ROOT / rel) != pin]
    report["task101_pins_ok"] = not bad
    if bad:
        raise hy.GuardError(f"G-frozen: TASK-101 pinned files differ: {bad}")
    path = ROOT / hy.MANIFEST
    if not path.exists():
        if gated:
            raise hy.GuardError("G-frozen: no manifest")
        report["manifest"] = None
        return
    mine = json.loads(path.read_text())
    differing = [
        rel for rel, pin in mine.get("files_sha256", {}).items() if sha256_file(ROOT / rel) != pin
    ]
    report["manifest"] = {"status": mine.get("status"), "files_differing": differing}
    if gated and (mine.get("status") != "FROZEN" or differing):
        raise hy.GuardError(f"G-frozen: status {mine.get('status')}, differing {differing}")


# ----- G_loc's training windows (§5.4) --
def corpus_check(corpus) -> dict:
    """``corpus.json`` against TASK-099's pin and every shard manifest against ``corpus.json``
    and TASK-099's manifest. Per-file hashes are checked as each file is read (``build_split``);
    no test-split file is opened."""
    from embodied_jepa import so101_wm as sw

    corpus = Path(corpus)
    got = sha256_file(corpus / "corpus.json")
    if got != sw.CORPUS_JSON_SHA256:
        raise hy.GuardError(f"G-corpus: corpus.json sha256 {got[:12]} is not TASK-099's")
    record = json.loads((corpus / "corpus.json").read_text())
    pins = json.loads((ROOT / sw.CORPUS_MANIFEST).read_text())["shard_manifest_sha256"]
    shards = {}
    for row in record["shards"]:
        name = f"shard-{int(row['shard']):02d}"
        h = sha256_file(corpus / name / "meta/jepa_manifest.json")
        pin = pins.get(name, pins.get(str(row["shard"])))
        if h != row["manifest_hash"] or h != pin:
            raise hy.GuardError(f"G-corpus: {name}'s manifest differs from its pins")
        manifest = json.loads((corpus / name / "meta/jepa_manifest.json").read_text())
        splits_rel = "meta/jepa_splits.json"
        if sha256_file(corpus / name / splits_rel) != manifest["sha256"][splits_rel]:
            raise hy.GuardError(f"G-corpus: {name}'s split file differs from its manifest")
        shards[name] = h
    return {"corpus_json_sha256": got, "shards": shards}


def feature_check(folder, splits) -> dict:
    """TASK-100's feature report against its recorded sha256, then the named splits' files
    (latents included) against the report. The test split is refused."""
    from embodied_jepa import so101_wm_data as sd

    folder = Path(folder)
    for split in splits:
        hy.check_split(split)
    got = sha256_file(folder / "report.json")
    if got != hy.FEATURE_REPORT_SHA256:
        raise hy.GuardError(f"G-features: report.json sha256 {got[:12]} is not TASK-100's")
    report = json.loads((folder / "report.json").read_text())
    try:
        files = sd.verify_store_files(folder, report["files_sha256"], splits)
    except ContractError as error:
        raise hy.GuardError(str(error)) from error
    return {"report_sha256": got, "files": files}


def _read_parquet_columns(path) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["observation.state", "action"])
    return {
        "state": np.asarray(table["observation.state"].to_pylist(), np.float32),
        "action": np.asarray(table["action"].to_pylist(), np.float32),
    }


def build_split(split: str, features, corpus, out, *, limit: int | None = None) -> dict:
    """§5.4's windows of one split (train or val; the test split is refused before any file is
    opened): for every episode of TASK-100's feature table, the corpus's per-attempt facts,
    phase labels, targets and proprioception (each file hash-checked against its shard
    manifest), the window rule, the store-corpus alignment guard on every window, and then the
    window frames' raw float16 latents gathered from the store (not featurised again).

    Writes ``<out>/{latents,x35,labels,index}_<split>.npy`` and ``windows_<split>.json``."""
    from embodied_jepa.so101_play_corpus import PHASES, REST_Z

    hy.check_split(split)
    features, corpus, out = Path(features), Path(corpus), Path(out)
    table = json.loads((features / f"episodes_{split}.json").read_text())
    if limit is not None:
        table = table[: int(limit)]
    actions = np.load(features / f"actions_{split}.npy")
    phase_store = np.load(features / f"phase_{split}.npy")
    latents = np.load(features / f"latents_{split}.npy", mmap_mode="r")
    descend, hold = PHASES.index("descend"), PHASES.index("hold")
    manifests, split_files = {}, {}
    index, x35, labels, windows = [], [], [], []
    excluded = {"not_grasped": 0, "early_close": 0, "no_lift_end": 0, "no_descend": 0}
    attempts_total = 0
    for row in table:
        shard, eid = row["shard"], row["episode_id"]
        offset, n = int(row["offset"]), int(row["frames"])
        if shard not in manifests:
            manifests[shard] = json.loads((corpus / shard / "meta/jepa_manifest.json").read_text())
            split_files[shard] = json.loads((corpus / shard / "meta/jepa_splits.json").read_text())[
                "episodes"
            ]
        if eid not in split_files[shard].get(split, []) or eid in split_files[shard].get(
            hy.REFUSED_SPLIT, []
        ):
            raise hy.GuardError(f"G-split: {eid} is not in the corpus's {split} split")
        mrow = next(r for r in manifests[shard]["episodes"] if r["episode_id"] == eid)
        meta = mrow["metadata"]
        parquet = corpus / shard / mrow["path"]
        if sha256_file(parquet) != manifests[shard]["sha256"][mrow["path"]]:
            raise hy.GuardError(f"G-corpus: {eid}'s episode file differs from its manifest")
        sidecar = corpus / shard / meta["sidecar"]
        if sha256_file(sidecar) != meta["sidecar_sha256"]:
            raise hy.GuardError(f"G-corpus: {eid}'s sidecar differs from its manifest")
        cols = _read_parquet_columns(parquet)
        with np.load(sidecar) as d:
            side = {k: d[k] for k in ("phase", "targets", "cube_position", "gripper_contact")}
        if not (len(cols["state"]) == len(side["phase"]) == n == mrow["length"]):
            raise hy.GuardError(f"G-corpus: {eid}'s lengths differ from the feature table")
        attempts = meta["facts"]["attempts"]
        attempts_total += len(attempts)
        w = hy.grasp_windows(side["phase"], attempts, n - 1, descend=descend, hold=hold)
        for k, v in w["excluded"].items():
            excluded[k] += v
        rise = side["cube_position"][:, 2].astype(np.float64) - REST_Z
        for win in w["windows"]:
            idx = np.arange(win["start"], win["end"])
            if not (
                np.array_equal(actions[offset + idx], cols["action"][idx])
                and np.array_equal(phase_store[offset + idx], side["phase"][idx])
            ):
                raise hy.GuardError(f"G-align: the store and the corpus differ on {eid}")
            index.append(offset + idx)
            x35.append(
                hy.nonlatent(
                    cols["state"][idx, :6],
                    cols["state"][idx, 6:],
                    side["targets"][idx],
                    idx - win["start"],
                )
            )
            labels.append(actions[offset + idx])
            windows.append(
                {
                    "episode_id": eid,
                    "shard": shard,
                    "seed": int(row["seed"]),
                    **win,
                    "commands": int(len(idx)),
                    "task_success_inside": hy.window_task_success(
                        rise, side["gripper_contact"], win["start"], win["end"]
                    ),
                }
            )
    index = np.concatenate(index).astype(np.int64)
    x35 = np.concatenate(x35).astype(np.float32)
    labels = np.concatenate(labels).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"{k}_{split}.npy" for k in ("latents", "x35", "labels", "index")}
    for p in (*paths.values(), out / f"windows_{split}.json"):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    cache = np.lib.format.open_memmap(
        paths["latents"], mode="w+", dtype=np.float16, shape=(len(index), hy.LATENT_DIM)
    )
    for lo in range(0, len(index), 4096):  # index is ascending: sequential reads of the store
        chunk = index[lo : lo + 4096]
        cache[lo : lo + len(chunk)] = np.asarray(latents[chunk]).reshape(len(chunk), -1)
    cache.flush()
    del cache
    np.save(paths["x35"], x35)
    np.save(paths["labels"], labels)
    np.save(paths["index"], index)
    (out / f"windows_{split}.json").write_text(json.dumps(windows))
    files = {p.name: sha256_file(p) for p in (*paths.values(), out / f"windows_{split}.json")}
    return {
        "split": split,
        "episodes": len(table),
        "attempts": attempts_total,
        "windows": len(windows),
        "frames": int(len(index)),
        "excluded": excluded,
        "windows_with_task_success_inside": int(sum(w["task_success_inside"] for w in windows)),
        "window_commands": {
            "min": int(min(w["commands"] for w in windows)),
            "median": float(np.median([w["commands"] for w in windows])),
            "max": int(max(w["commands"] for w in windows)),
        },
        "holds_median": float(np.median([w["holds"] for w in windows])),
        "files_sha256": files,
    }


def load_split_frames(folder, split: str, latent_moments, *, in_memory: bool = True):
    """A windows split as ``so101_grasp_policy.Frames`` (hash-checked against the windows
    report)."""
    from embodied_jepa import so101_grasp_policy as gp

    hy.check_split(split)
    folder = Path(folder)
    report = json.loads((folder / "windows_report.json").read_text())
    for name, want in report["splits"][split]["files_sha256"].items():
        if sha256_file(folder / name) != want:
            raise hy.GuardError(f"G-windows: {name} differs from the windows report")
    raw16 = np.load(folder / f"latents_{split}.npy", mmap_mode=None if in_memory else "r")
    return gp.Frames(
        raw16,
        np.load(folder / f"x35_{split}.npy"),
        np.load(folder / f"labels_{split}.npy"),
        latent_moments,
    )


# ----- the privileged aims (S+G's scripted approach, W+F's follower; §6) --
def follower_aims(sim, home, config=None) -> dict:
    """Follower v1's grasp aims on the reset scene, exactly as ``so101_followers._pick`` (and
    the envelope's ``perturbed_lift``) computes them before its pre-grasp drive: the closing
    axis, the pre-grasp and grasp fingertip points (zero grasp offset) and q_pre (IK from
    ``home``). Reads simulator truth; the runner calls it at the reset, outside the sentinel."""
    from embodied_jepa import so101_followers as f
    from embodied_jepa.so101_scripted import JAW_CENTRE_OFFSET_M, _closing_axis, solve_ik

    cfg = f.FOLLOWER_V1 if config is None else config
    truth = sim.task_truth()
    cube = truth["cube_position"]
    closing, _ = f._grasp_frame(cube, truth["cube_quaternion"])
    shift = JAW_CENTRE_OFFSET_M - cfg.fixed_jaw_clearance_m
    q_home = np.asarray(home[:5], float).copy()

    def aim(height):
        point = cube + np.array([0.0, 0.0, height])
        q, _ = solve_ik(sim, q_home, point, closing)
        return point + _closing_axis(sim, q) * shift

    pre, grasp = aim(cfg.pregrasp_height_m), aim(0.0)
    q_pre, _ = solve_ik(sim, q_home, pre, closing)
    return {"closing": closing, "pre": pre, "grasp": grasp, "q_pre": q_pre, "config": cfg}


def drive_action(q_goal, targets, scales, gripper) -> np.ndarray:
    """Follower v1's drive law (``_Runner.drive`` and its float32 cast in ``step``)."""
    delta = (np.asarray(q_goal, float) - np.asarray(targets, float)[:5]) / np.asarray(
        scales
    ).astype(float)
    return np.asarray(np.concatenate((np.clip(delta, -1, 1), [gripper])), dtype=np.float32)


class ScriptedApproach:
    """S+G's approach (§6, §7.4): follower v1's pre-grasp drive toward ``q_pre + δ`` with the jaw
    open, for 40 commands (the envelope's "pre" construction). With δ = 0 it is the goal
    demonstration's first 40 commands (a guard checks it)."""

    def __init__(self, q_goal, scales):
        self.q_goal = np.asarray(q_goal, float)
        self.scales = np.asarray(scales, np.float32)

    def act(self, targets) -> np.ndarray:
        return drive_action(self.q_goal, targets, self.scales, 1.0)


class FollowerTail:
    """W+F's phase 2 (§6): follower v1 from the hand-over pose -- its descent waypoints (lazy IK
    from q_pre, as ``_Runner.through``), the close, the lift and the hold, aimed from the reset's
    simulator truth (the envelope's "pre" path). Handed over at the demonstration's exact
    command-40 state it reproduces the demonstration's commands 41 onward (guard G-zero)."""

    def __init__(self, sim, aims: dict, scales, hold_action):
        self.sim, self.aims = sim, aims
        self.scales = np.asarray(scales, np.float32)
        self.hold_action = hold_action
        self.q = np.asarray(aims["q_pre"], float).copy()
        self.targets = None
        self._schedule = self._steps()

    def _ik(self, point):
        from embodied_jepa.so101_scripted import solve_ik

        q, _ = solve_ik(self.sim, self.q, point, self.aims["closing"])
        self.q = q
        return q

    def _through(self, start, end, waypoints, gripper, steps):
        per = max(1, steps // waypoints)
        for k in range(1, waypoints + 1):
            q = self._ik(start + (end - start) * k / waypoints)
            for _ in range(per):
                yield drive_action(q, self.targets, self.scales, gripper)

    def _steps(self):
        cfg, pre, grasp = self.aims["config"], self.aims["pre"], self.aims["grasp"]
        yield from self._through(pre, grasp, cfg.descend_waypoints, 1.0, cfg.descend_steps)
        q_close = self.q.copy()
        for _ in range(cfg.close_steps):
            yield drive_action(q_close, self.targets, self.scales, -1.0)
        lifted = grasp + np.array([0.0, 0.0, cfg.lift_height_m])
        yield from self._through(grasp, lifted, 3, -1.0, cfg.lift_steps)
        while True:
            yield np.asarray(self.hold_action(), np.float32)

    def act(self, targets) -> np.ndarray:
        self.targets = np.asarray(targets, float).copy()
        return next(self._schedule)


# ----- phase 1 and phase 2 controllers (§5, §6) --
class Phase1CEM:
    """TASK-101's W, unchanged in every planner setting, aimed at subgoal 1 only."""

    def __init__(self, *, adapter, horizon: int, goal_latent, params, seed: int):
        from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
        from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner

        self.adapter, self.goal, self.params = adapter, np.asarray(goal_latent), params
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

    def act(self, z, targets) -> tuple[np.ndarray, dict]:
        import torch

        t0 = time.perf_counter()
        plan = self.planner.plan(
            self.adapter, z, self.goal, projector=rt.Projector(self.params, targets)
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        if not np.isfinite(plan.costs).all():
            raise hy.GuardError("G-finite: a non-finite planned cost")
        return np.asarray(plan.requested_actions[0, 0], np.float32).copy(), {
            "projected": np.asarray(plan.actions[0, 0], np.float32).copy(),
            "cost": float(plan.costs[0]),
            "plan_seconds": seconds,
        }


class Phase1Pose:
    """H-pose+G's approach: TASK-101's non-learned pose follower toward subgoal 1's joints."""

    def __init__(self, goal_q, scales):
        self.goal = np.asarray(goal_q, np.float64)
        self.scales = np.asarray(scales, np.float32)

    def act(self, targets) -> np.ndarray:
        return cl.pose_action(targets, self.goal, self.scales)


class HybridController:
    """One episode's controller: phase 1 (``cem``, ``pose``, ``scripted`` or none), the
    hand-over (``HandoverRule``, a fixed command for S+G, or none) and phase 2 (G_loc,
    W+F's follower or none). ``act`` returns ``None`` when phase 2 is none and the hand-over
    fired (A's approach episodes end there)."""

    def __init__(
        self,
        *,
        featuriser,
        phase1=None,
        phase1_kind=None,
        rule: hy.HandoverRule | None = None,
        fixed_handover: int | None = None,
        gloc=None,
        follower: FollowerTail | None = None,
    ):
        self.featuriser = featuriser
        self.phase1, self.phase1_kind = phase1, phase1_kind
        self.rule, self.fixed = rule, fixed_handover
        self.gloc, self.follower = gloc, follower
        self.command = 0
        self.handover_command = None if phase1 is not None else 0
        self.event = None

    def _phase(self, q) -> int:
        if self.phase1 is None:
            return 2
        if self.rule is not None:
            return self.rule.update(q)
        if self.command >= self.fixed:
            if self.event is None:
                self.event = {
                    "command": self.command,
                    "how": "fixed",
                    "phase1_commands": self.fixed,
                }
            return 2
        return 1

    def act(self, frames, q, qvel, targets):
        phase = self._phase(q)
        info: dict = {"phase": phase}
        if phase == 2 and self.handover_command is None:
            self.handover_command = self.command
        needs_latent = (phase == 1 and self.phase1_kind == "cem") or (
            phase == 2 and self.gloc is not None
        )
        z = None
        if needs_latent:
            t0 = time.perf_counter()
            raw16, zz = self.featuriser.encode({c: frames[c][None] for c in cl.CAMERAS})
            info["raw16"], z = raw16[0], zz[0]
            info["encode_seconds"] = time.perf_counter() - t0
        self.command += 1
        if phase == 1:
            if self.phase1_kind == "cem":
                action, extra = self.phase1.act(z, targets)
                info |= extra
            else:
                action = self.phase1.act(targets)
            return action, info
        if self.gloc is not None:
            t = self.command - 1 - self.handover_command
            t0 = time.perf_counter()
            action, raw = self.gloc.act(z, q, qvel, targets, t)
            info |= {"gloc_raw": raw, "policy_seconds": time.perf_counter() - t0, "clock": t}
            return action, info
        if self.follower is not None:
            return self.follower.act(targets), info
        return None, info


class RandController:
    def __init__(self, seed: int):
        self.rng = hy.rand_rng(seed)

    def act(self, frames, q, qvel, targets):
        return cl.rand_action(self.rng), {"phase": 0}


class HoldController:
    def __init__(self, hold_action):
        self.hold_action = hold_action

    def act(self, frames, q, qvel, targets):
        return np.asarray(self.hold_action(), np.float32), {"phase": 0}


def make_controller(spec: dict, ctx: dict, robot, params, aims: dict | None):
    arm, seed, slot = spec["arm"], int(spec["seed"]), int(spec["slot"])
    if arm == "rand":
        return RandController(seed)
    if arm == "hold":
        return HoldController(robot.hold_action)
    a = ctx["goals"]["arrays"]
    if int(a["seeds"][slot]) != seed:
        raise hy.GuardError("G-goals: the slot's goal belongs to another reset")
    goal_q1 = np.asarray(a["joints"][slot, hy.SUBGOAL_1], np.float64)
    p1, handover, p2 = hy.ARM_PLAN[arm]
    phase1 = rule = fixed = None
    kind = None
    if p1 is not None and p1.startswith("cem"):
        h, cost = cl.parse_config(spec["config"])
        model = hy.PRIMARY_J if p1 == "cem:J" else hy.TWIN_J
        phase1 = Phase1CEM(
            adapter=ctx["adapter"](model, cost),
            horizon=h,
            goal_latent=ctx["goal_latents"][slot, hy.SUBGOAL_1],
            params=params,
            seed=hy.planner_seed(seed),
        )
        kind = "cem"
    elif p1 == "pose":
        phase1, kind = Phase1Pose(goal_q1, robot.scales), "pose"
    elif p1 == "scripted":
        delta = hy.p_offset(seed, float(spec.get("level", 0.0)))
        phase1, kind = ScriptedApproach(aims["q_pre"] + delta, robot.scales), "scripted"
    if handover == "rule":
        rule = hy.HandoverRule(goal_q1, float(spec["tau_h"]))
    elif handover == "fixed40":
        fixed = hy.SCRIPTED_APPROACH_COMMANDS
    gloc = follower = None
    if p2 is not None and p2.startswith("gloc"):
        gloc = ctx["gloc"](p2.split(":")[1])
    elif p2 == "follower":
        follower = FollowerTail(robot.sim, aims, robot.scales, robot.hold_action)
    return HybridController(
        featuriser=ctx.get("featuriser"),
        phase1=phase1,
        phase1_kind=kind,
        rule=rule,
        fixed_handover=fixed,
        gloc=gloc,
        follower=follower,
    )


# ----- the episode loop (§2, §6) --
STATUS_CODES = rt.STATUS_CODES
LOG_KEYS = (
    "phase",
    "requested",
    "applied",
    "projected",
    "gloc_raw",
    "status",
    "cost",
    "plan_seconds",
    "encode_seconds",
    "policy_seconds",
    "q",
    "qvel",
    "targets",
    "cube",
    "quat",
    "site",
    "site_xmat",
    "contacts",
    "encoded",
    "step_seconds",
)
NAN6 = np.full(6, np.nan, np.float32)


def run_episode(spec: dict, ctx: dict) -> dict:
    """One arm on one reset. ``spec``: arm, seed, slot, config, tau_h, level, keep_frames,
    output. Writes ``<output>/<seed>.npz`` (per-step logs; frames only when ``keep_frames``)
    and returns the episode summary."""
    arm, seed, slot = spec["arm"], int(spec["seed"]), int(spec["slot"])
    hy.check_arm(arm)
    out_dir = Path(spec["output"])
    path = out_dir / f"{seed}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    keep = bool(spec.get("keep_frames"))
    started = time.perf_counter()
    goals = ctx["goals"]
    if arm == "F":
        r = rt.run_follower(seed, slot, goals, path, render=keep, started=started)
        return r | {"level": None, "tau_h": None}
    gpu_arm = arm in hy.GPU_ARMS
    render = gpu_arm or keep
    guards = {}
    if arm == "W+F":
        guards["g_zero_follower"] = g_zero_follower(seed, slot, goals)
        if not guards["g_zero_follower"]:
            raise hy.GuardError(f"G-zero: the follower-from-hand-over path diverged on {seed}")
    robot = rt.make_robot(render)
    task = rt.lift_task(robot)
    task.reset(seed)
    aims = follower_aims(robot.sim, robot.home) if arm in ("S+G", "W+F") else None
    sentinel = rt.TruthSentinel(robot.sim)
    params = cl.PreviewParams.of(robot)
    controller = make_controller(spec, ctx, robot, params, aims)
    rows: dict[str, list] = {k: [] for k in LOG_KEYS}
    latents, frames, reasons = [], [], []
    termination, score = "running", None
    ended_at_handover = False
    try:
        for _ in range(hy.MAX_COMMANDS + 1):
            t_step = time.perf_counter()
            obs = robot.observe()
            imgs = {c: obs.images[c][0] for c in cl.CAMERAS}
            q = np.asarray(obs.state.values[0, : cl.JOINTS], np.float64)
            qvel = np.asarray(obs.state.values[0, cl.JOINTS :], np.float64)
            targets = robot.sim.targets.copy()
            with sentinel.controller():
                action, info = controller.act(imgs, q, qvel, targets)
            if action is None:  # A's approach episodes end at the hand-over
                ended_at_handover = True
                termination = "handover"
                break
            if keep:
                frames.append(np.stack([imgs[c] for c in cl.CAMERAS]))
            if "raw16" in info:
                latents.append(info["raw16"])
            action = np.asarray(action, np.float32)
            if not np.isfinite(action).all():
                raise hy.GuardError("G-finite: a non-finite action")
            result = robot.execute(action)
            if result.status != "stopped" and "projected" in info:
                if not np.array_equal(result.applied_action, info["projected"]):
                    raise hy.GuardError(
                        "G-projection: the executed action differs from the scored one"
                    )
            truth = rt.truth_row(robot.sim)
            rows["phase"].append(info.get("phase", 0))
            rows["requested"].append(action)
            rows["applied"].append(
                NAN6
                if result.applied_action is None
                else np.asarray(result.applied_action, np.float32)
            )
            rows["projected"].append(info.get("projected", NAN6))
            rows["gloc_raw"].append(np.asarray(info.get("gloc_raw", NAN6), np.float32))
            rows["status"].append(STATUS_CODES[result.status])
            for key in ("cost", "plan_seconds", "encode_seconds", "policy_seconds"):
                rows[key].append(info.get(key, np.nan))
            rows["q"].append(q)
            rows["qvel"].append(qvel)
            rows["targets"].append(targets)
            for key in ("cube", "quat", "site", "site_xmat", "contacts"):
                rows[key].append(truth[key])
            rows["encoded"].append("raw16" in info)
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
            raise hy.GuardError("G-steps: the episode outran the task's command budget")
        if keep and not ended_at_handover:
            obs = robot.observe()
            frames.append(np.stack([obs.images[c][0] for c in cl.CAMERAS]))
    finally:
        robot.close()
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    if latents:
        arrays["latents_raw16"] = np.asarray(latents, np.float16)
    if frames:
        arrays["frames"] = np.asarray(frames, np.uint8)
    np.savez_compressed(path, **arrays) if frames else np.savez(path, **arrays)
    if arm == "S+G" and float(spec.get("level", 0.0)) == 0.0:
        demo = goals["arrays"]["demo_actions"][slot, : hy.SCRIPTED_APPROACH_COMMANDS]
        guards["g_zero_approach"] = (
            bool(
                len(arrays["applied"]) >= hy.SCRIPTED_APPROACH_COMMANDS
                and np.array_equal(arrays["applied"][: hy.SCRIPTED_APPROACH_COMMANDS], demo)
            )
            or bool(  # an episode that ends inside the approach compares what ran
                len(arrays["applied"]) < hy.SCRIPTED_APPROACH_COMMANDS
                and np.array_equal(arrays["applied"], demo[: len(arrays["applied"])])
            )
        )
        if not guards["g_zero_approach"]:
            raise hy.GuardError(f"G-zero: S+G's approach differs from the demonstration on {seed}")
    goal_q1 = (
        None
        if arm in ("rand", "hold")
        else np.asarray(goals["arrays"]["joints"][slot, hy.SUBGOAL_1], np.float64)
    )
    summary = summarise(
        spec, arrays, controller, score, termination, reasons, started, path, goal_q1
    )
    return summary | {"guards": guards}


def _phase_seconds(arrays, phase) -> dict:
    s = arrays["step_seconds"][arrays["phase"] == phase] if len(arrays["phase"]) else []
    if len(s) == 0:
        return {"commands": 0, "median": None, "p95": None, "max": None}
    return {
        "commands": int(len(s)),
        "median": float(np.median(s)),
        "p95": float(np.percentile(s, 95)),
        "max": float(np.max(s)),
    }


def summarise(spec, arrays, controller, score, termination, reasons, started, path, goal_q1):
    rule = getattr(controller, "rule", None)
    handover = None
    if rule is not None:
        handover = rule.event
    elif getattr(controller, "event", None) is not None:  # S+G's fixed hand-over at command 40
        handover = dict(controller.event)
        k = handover["command"]
        if goal_q1 is not None and len(arrays["q"]) > k:
            handover["arm_rms_rad"] = hy.arm_rms(arrays["q"][k], goal_q1)
            handover["jaw_error_rad"] = float(arrays["q"][k][hy.JAW] - goal_q1[hy.JAW])
    status = arrays["status"]
    return {
        "arm": spec["arm"],
        "seed": int(spec["seed"]),
        "slot": int(spec["slot"]),
        "config": spec.get("config"),
        "tau_h": spec.get("tau_h"),
        "level": spec.get("level"),
        "success": bool(score and score["success"]),
        "termination": termination,
        "success_step": score["success_step"] if score else None,
        "max_lift_m": score["max_lift_m"] if score else None,
        "longest_hold_steps": score["longest_hold_steps"] if score else None,
        "commands": int(len(status)),
        "stopped": termination.startswith("stopped"),
        "clipped_commands": int(np.sum(status == 1)) if len(status) else 0,
        "reasons": sorted(set(reasons)),
        "handover": handover,
        "closest_rms_to_subgoal1": None if rule is None else float(rule.closest),
        "phase1": _phase_seconds(arrays, 1),
        "phase2": _phase_seconds(arrays, 2),
        "plan_seconds_median": (
            float(np.nanmedian(arrays["plan_seconds"]))
            if len(arrays["plan_seconds"]) and np.isfinite(arrays["plan_seconds"]).any()
            else None
        ),
        "step_seconds_max": float(np.max(arrays["step_seconds"])) if len(status) else None,
        "episode_seconds": time.perf_counter() - started,
        "file": path.name,
        "file_sha256": sha256_file(path),
        "frames_kept": "frames" in arrays,
        "pid": os.getpid(),
    }


# ----- G-zero for the follower-from-hand-over path (§7.1, §12) --
def g_zero_follower(seed: int, slot: int, goals: dict) -> bool:
    """On a fresh, unrendered simulation of ``seed``: S+G's scripted approach with δ = 0 (the
    demonstration's first 40 commands), the hand-over at that exact command-40 state to
    ``FollowerTail``, run until the task ends; every executed command must equal the
    demonstration's bit for bit, with the same outcome."""
    a = goals["arrays"]
    demo = a["demo_actions"][slot, : int(a["demo_lengths"][slot])]
    report = goals["report"]["demonstrations"][str(seed)]
    robot = rt.make_robot(False)
    task = rt.lift_task(robot)
    applied = []
    score = None
    try:
        task.reset(seed)
        aims = follower_aims(robot.sim, robot.home)
        approach = ScriptedApproach(aims["q_pre"], robot.scales)
        tail = FollowerTail(robot.sim, aims, robot.scales, robot.hold_action)
        for n in range(hy.MAX_COMMANDS + 1):
            robot.observe()
            targets = robot.sim.targets.copy()
            if n < hy.SCRIPTED_APPROACH_COMMANDS:
                action = approach.act(targets)
            else:
                action = tail.act(targets)
            result = robot.execute(action)
            if result.status == "stopped":
                return False
            applied.append(np.asarray(result.applied_action, np.float32))
            score = task.evaluate()
            if score["done"]:
                break
    finally:
        robot.close()
    applied = np.asarray(applied, np.float32)
    return bool(
        applied.shape == demo.shape
        and np.array_equal(applied, demo)
        and bool(score["success"]) == bool(report["success"])
        and score["success_step"] == report["success_step"]
    )


# ----- G_loc in the workers --
def load_gloc_selection(path) -> dict:
    """``gloc.json``: the primary seed, the replication seeds and every checkpoint's path and
    sha256 (written by ``train_task102.py select``)."""
    sel = json.loads(Path(path).read_text())
    if sel.get("task") != hy.TASK or sel.get("primary") is None:
        raise hy.GuardError("G-gloc: no primary G_loc in the selection file")
    return sel


def gloc_loader(selection: dict, device: str):
    from embodied_jepa import so101_grasp_policy as gp

    cache: dict = {}

    def get(role: str):
        if role == "primary":
            name = selection["primary"]
        elif role.startswith("replication"):
            name = selection["replication"][int(role.removeprefix("replication"))]
        else:
            raise ContractError(f"unknown G_loc role {role!r}")
        if name not in cache:
            entry = selection["checkpoints"][name]
            cache[name] = gp.load_checkpoint(entry["path"], device, entry["sha256"])
        return cache[name]

    return get


# ----- the worker pool (§13) --
_CTX: dict = {}


def init_worker(goals_dir, goals_sha256, device, gloc_file, threads: int = 2) -> None:
    """Each worker: its own CUDA context, deterministic setup, featuriser, models and G_loc
    (lazily)."""
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
    models: dict = {}

    def adapter(name, cost):
        if name not in models:
            models[name] = rt.load_model(name, device)[0]
        return rt.LatentCost(models[name], cost, action_blind=name == hy.TWIN_J)

    _CTX["adapter"] = adapter
    if gloc_file:
        _CTX["gloc"] = gloc_loader(load_gloc_selection(gloc_file), device)


def _worker_episode(spec: dict) -> dict:
    return run_episode(spec, _CTX)


def run_pool(specs, *, goals_dir, goals_sha256, device, workers, gloc_file, on_result=None):
    """Episodes in a spawn pool, one fresh worker process per episode (TASK-101's form, because
    MuJoCo's EGL renderer is not bit-stable across a process's history)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        max(1, int(workers)),
        initializer=init_worker,
        initargs=(str(goals_dir), goals_sha256, device, gloc_file),
        maxtasksperchild=1,
    ) as pool:
        out = []
        for r in pool.imap(_worker_episode, specs, chunksize=1):
            out.append(r)
            if on_result:
                on_result(r)
        return out


# ----- the determinism pairs (§7.3.5; TASK-101 7.2.4's conditional form) --
ACTION_KEYS = ("applied", "requested", "status", "phase")


def _first_difference(x, y):
    n = min(len(x), len(y))
    a = np.asarray(x[:n]).reshape(n, -1)
    b = np.asarray(y[:n]).reshape(n, -1)
    same = np.all((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b, 1)
    bad = np.flatnonzero(~same)
    if len(bad):
        return int(bad[0])
    return None if len(x) == len(y) else n


def pair_divergence(path_a, path_b, ra: dict, rb: dict) -> dict:
    """Where two runs of one episode first differ: in the live latent and in the executed
    actions (step index of the episode), and whether the outcome agrees. Passes the conditional
    form when the actions never diverge before the latent does."""
    with np.load(path_a) as x, np.load(path_b) as y:
        latent = None
        if "latents_raw16" in x.files and "latents_raw16" in y.files:
            k = _first_difference(x["latents_raw16"], y["latents_raw16"])
            if k is not None:  # the step at which the k-th encoded latent was taken
                steps = [
                    int(s[k])
                    for s in (np.flatnonzero(x["encoded"]), np.flatnonzero(y["encoded"]))
                    if k < len(s)
                ]
                latent = min(steps) if steps else int(min(len(x["encoded"]), len(y["encoded"])))
        actions = [_first_difference(x[k], y[k]) for k in ACTION_KEYS]
    acts = [a for a in actions if a is not None]
    action = min(acts) if acts else None
    keys = ("success", "termination", "success_step", "commands")
    same_outcome = all(ra[k] == rb[k] for k in keys)
    identical = latent is None and action is None and same_outcome
    conditional = (action is None and same_outcome) or (
        latent is not None and action is not None and action >= latent
    )
    return {
        "first_latent_difference": latent,
        "first_action_difference": action,
        "same_outcome": same_outcome,
        "bit_identical": identical,
        "conditional_determinism": bool(conditional),
    }


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
