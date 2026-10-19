"""TASK-101's runner (protocol docs/experiments/so101_closed_loop_lift_v1.md §7, §8, §12, §18).

Every job runs ``so101_cube_lift_v1`` (amendment A1, §18.2). Subcommands (each refuses an
existing output):

* ``goals``    -- the cohort's goal demonstrations (follower v1, rendered), §4's replacement
  from the reserve, the subgoal frames encoded by the live latent, the hashed goal store, and
  nothing else (A1, §18.5 (i): the CPU arms' evidence episodes run in the cohort's episode jobs).
* ``episodes`` -- arms on a cohort's kept resets, one pool of workers (GPU for CEM arms; CPU arms
  run with ``--device none``). For K' and D' it refuses, before any controller, simulator or seed
  is constructed, unless the check ledger admits it (§18.5's ordering rule, R28.70).
* ``check``    -- §7.2's checks, each appended to the ledger: ``scene`` (7.2.0), ``latent``
  (7.2.1), ``renc`` (7.2.2, per cohort), ``projector`` (7.2.3), ``determinism`` (7.2.4,
  conditional form), ``inert`` (7.2.6); reported: ``ties``, ``adapter``, ``render``, ``marker``.
* ``delta``    -- §18.6's S0-NULL diagnostic on cohort X's goal store (planning queried, not
  executed).

Stage 0' runs on K', D', X and debug seeds (K and D are spent); S refuses until the manifest is
FROZEN (``--gated``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_closed_loop_runtime as rt


def gpu_prelude(report: dict, device: str) -> None:
    from embodied_jepa import devices
    from embodied_jepa import run_tools as tools

    if device == "cuda":
        tools.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")


def guards(report: dict, gated: bool) -> None:
    if gated and rt.tree_dirty():
        raise cl.GuardError("G-tree: the tracked tree is dirty")
    rt.check_frozen_task100(report)
    rt.check_manifest(report, strict=gated)


def ledger_path(args) -> Path:
    """§18.5 (2): the ledger lives in the run's output directory (the stage root, the parent of
    every job's output folder) unless ``--ledger`` names it."""
    return Path(args.ledger) if args.ledger else Path(args.output).parent / cl.LEDGER_FILE


def require_v1(goals: dict, cohorts) -> None:
    """A check reads a goal store of the named cohorts, made on ``so101_cube_lift_v1``."""
    if goals["report"]["cohort"] not in cohorts:
        raise cl.GuardError(f"this check reads a goal store of {cohorts}")
    if goals["report"].get("task_name") != cl.LIFT_TASK_NAME:
        raise cl.GuardError(f"G-goals: the goal store was not made on {cl.LIFT_TASK_NAME}")


def check_cohort(name: str, gated: bool) -> None:
    cl.check_cohort_unspent(name)
    if name in cl.GATED_COHORTS and not gated:
        raise cl.GuardError(
            "G-gated: the gated cohort runs only from the frozen revision (--gated)"
        )
    if name not in cl.GATED_COHORTS and gated:
        raise cl.GuardError("G-gated: --gated is for cohort S only")


# ----- goals --
def cmd_goals(args) -> int:
    out = Path(args.output)
    check_cohort(args.cohort, args.gated)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    report = rt.header("goals", cohort=args.cohort, task_name=cl.LIFT_TASK_NAME)
    guards(report, args.gated)
    gpu_prelude(report, args.device)
    featuriser = rt.LiveFeaturiser(args.device)
    report["fit_report_sha256"] = featuriser.fit_report_sha256
    report["encoder_digest"] = featuriser.featuriser.digest
    demos: dict = {}

    def demonstrate(seed):
        cl.check_seed(args.cohort, seed)
        d = rt.demonstrate(seed, render=True)
        demos[seed] = d
        rt.log(f"demo {seed}: {d['termination_reason']} step {d['success_step']} kept={d['kept']}")
        return d["kept"], d["reason"], None

    limit = args.limit
    if limit is not None:  # debug plumbing only: the first `limit` seeds of the debug cohort
        if args.cohort != "debug":
            raise cl.GuardError("--limit is for the debug cohort only")
        cl.COHORTS["debug"] = cl.Cohort("debug", cl.COHORTS["debug"].first, limit, None)
    kept = cl.keep_cohort(args.cohort, demonstrate)
    seeds = kept["kept"]
    n = len(seeds)
    frames = np.zeros((n, cl.SUBGOALS, 2, cl.IMAGE_SIZE, cl.IMAGE_SIZE, 3), np.uint8)
    raw = np.zeros((n, cl.SUBGOALS, cl.TOKENS, cl.WIDTH), np.float16)
    joints = np.zeros((n, cl.SUBGOALS, cl.JOINTS))
    cube = np.zeros((n, cl.SUBGOALS, 3))
    site = np.zeros((n, cl.SUBGOALS, 3))
    xmat = np.zeros((n, cl.SUBGOALS, 9))
    lengths = np.array([demos[s]["commands"] for s in seeds], np.int64)
    actions = np.full((n, int(lengths.max()), cl.JOINTS), np.nan, np.float32)
    for i, s in enumerate(seeds):
        d = demos[s]
        caps = d["captured"]
        if [c["command"] for c in caps] != list(cl.SUBGOAL_COMMANDS):
            raise cl.GuardError(f"G-goals: {s}'s captures are not the four subgoals")
        for k, c in enumerate(caps):
            frames[i, k] = c["frames"]
            joints[i, k] = c["joints"]
            cube[i, k] = c["cube_cm"]
            site[i, k] = c["site"]
            xmat[i, k] = c["site_xmat"]
        raw[i] = featuriser.raw16({cam: frames[i, :, j] for j, cam in enumerate(cl.CAMERAS)})
        actions[i, : lengths[i]] = d["applied"]
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
    report["kept"] = seeds
    report["rejected"] = kept["rejected"]
    report["drawn"] = kept["drawn"]
    report["demonstrations"] = {
        str(s): {
            k: d[k]
            for k in (
                "success",
                "termination_reason",
                "success_step",
                "stopped",
                "commands",
                "kept",
                "reason",
            )
        }
        for s, d in demos.items()
    }
    report["goals_seconds"] = time.monotonic() - t0
    sha = cl.save_goal_store(out, arrays, report)
    rt.log(f"goal store {sha[:12]}: kept {n}, rejected {len(kept['rejected'])}")
    report2 = {
        "goals_sha256": sha,
        "seconds_total": time.monotonic() - t0,
        "finished_utc": rt.utc(),
    }
    rt.write_json(out / "job.json", report2)
    return 0


# ----- episodes --
def parse_slots(text: str | None, n: int) -> list[int]:
    if not text:
        return list(range(n))
    lo, hi = (int(x) for x in text.split(":"))
    return list(range(lo, min(hi, n)))


def cmd_episodes(args) -> int:
    out = Path(args.output)
    check_cohort(args.cohort, args.gated)
    if out.exists():
        raise cl.GuardError(f"G-twice: {out} exists; a cohort and arm run once")
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    if goals["report"]["cohort"] != args.cohort:
        raise cl.GuardError("G-goals: the goal store belongs to another cohort")
    if goals["report"].get("task_name") != cl.LIFT_TASK_NAME:
        raise cl.GuardError(f"G-goals: the goal store was not made on {cl.LIFT_TASK_NAME}")
    verdict = None
    if args.cohort in cl.LEDGER_COHORTS:  # §18.5 (2): before any controller, simulator or seed
        if rt.tree_dirty():
            raise cl.GuardError("G-ledger: the tracked tree is dirty")
        verdict = cl.require_ledger(
            ledger_path(args), revision=rt.git("rev-parse", "HEAD"), goals_sha256=goals["sha256"]
        )
    out.mkdir(parents=True)
    t0 = time.monotonic()
    report = rt.header(
        "episodes",
        cohort=args.cohort,
        arms=args.arms,
        configs=args.configs,
        workers=args.workers,
        goals_sha256=goals["sha256"],
        slots=args.slots,
        evidence=args.evidence,
        device=args.device,
        task_name=cl.LIFT_TASK_NAME,
        ledger=None if verdict is None else str(ledger_path(args)),
        ledger_verdict=verdict,
    )
    guards(report, args.gated)
    if args.device == "cuda":
        gpu_prelude(report, args.device)
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    for s in seeds:
        cl.check_seed(args.cohort, s)
    slots = parse_slots(args.slots, len(seeds))
    arms = args.arms.split(",")
    configs = args.configs.split(",") if args.configs else [None]
    specs = []
    for slot in slots:
        for arm in arms:
            cl.check_arm(arm)
            cem = arm in cl.CEM_ARMS or arm == cl.DIAGNOSTIC
            if cem and args.device != "cuda":
                raise cl.GuardError("CEM arms run on the GPU")
            for config in configs if cem else [None]:
                name = arm if config is None else f"{arm}@{config}"
                folder = out / name
                folder.mkdir(exist_ok=True)
                specs.append(
                    {
                        "arm": arm,
                        "seed": seeds[slot],
                        "slot": slot,
                        "config": config,
                        "keep_frames": slot < args.evidence,
                        "output": str(folder),
                    }
                )
    report["episodes_planned"] = len(specs)
    done = []
    progress = out / "progress.jsonl"

    def on_result(r):
        done.append(r)
        with progress.open("a") as f:
            f.write(json.dumps(r, default=cl._json_default) + "\n")
        rt.log(
            f"{r['arm']}@{r['config']} {r['seed']}: {r['termination']} "
            f"({r['commands']} cmds, {r['episode_seconds']:.0f} s) [{len(done)}/{len(specs)}]"
        )

    results = rt.run_pool(
        specs,
        goals_dir=args.goals,
        goals_sha256=goals["sha256"],
        device=args.device,
        workers=args.workers,
        on_result=on_result,
    )
    report["episodes"] = results
    report["wall_seconds"] = time.monotonic() - t0
    rt.check_frozen_task100(report)
    if args.gated and rt.tree_dirty():
        raise cl.GuardError("G-tree: the tracked tree became dirty")
    report["finished_utc"] = rt.utc()
    rt.write_json(out / "report.json", report)
    return 0


# ----- checks (§7.2) --
def check_latent(args, report) -> dict:
    """§7.2.1: 64 stored frames per camera from 8 val episodes, encoded live (one frame per camera
    per call, as the controller does), equal to the stored float16 within 2 float16 ulps.

    The frames are 8 per episode, evenly spaced over the frames the store encoded in full
    batches of 64 (TASK-100's featuriser encoded each episode's last 401 mod 64 = 17 frames in a
    smaller batch, whose GPU arithmetic a live batch of 64 cannot reproduce; Stage 0 record)."""
    from embodied_jepa import so101_wm_data as sd

    feat = rt.LiveFeaturiser(args.device)
    shared = rt.shared_root()
    folder = shared / cl.TASK100_FEATURES
    table = json.loads((folder / "episodes_val.json").read_text())
    latents = np.load(folder / "latents_val.npy", mmap_mode="r")
    worst, count, rows = 0.0, 0, []
    for row in table[:8]:
        ep = sd.load_episode(shared / "data/so101-play-v1", row["shard"], row["episode_id"])
        n = row["frames"]
        last_full = (n // feat.featuriser.batch) * feat.featuriser.batch - 1
        for f in np.linspace(0, last_full, 8).astype(int):
            live = feat.raw16({c: ep["frames"][c][f][None] for c in cl.CAMERAS})[0]
            stored = np.asarray(latents[row["offset"] + f])
            ulp = np.spacing(np.abs(stored)).astype(np.float32)
            diff = np.abs(live.astype(np.float32) - stored.astype(np.float32))
            ratio = float((diff / ulp).max())
            worst = max(worst, ratio)
            count += 1
            rows.append(
                {
                    "episode": row["episode_id"],
                    "frame": int(f),
                    "max_ulps": ratio,
                    "bit_identical": bool(np.array_equal(live, stored)),
                    "exact_fraction": float((diff == 0).mean()),
                }
            )
    return {
        "frames": count,
        "max_ulps": worst,
        "bit_identical_frames": sum(r["bit_identical"] for r in rows),
        "passes": bool(worst <= 2.0),
        "rows": rows,
    }


def check_renc(args, report) -> dict:
    """§7.2.2, per cohort (A1, §18.5): TASK-100's pretrained R-enc on the live-encoded subgoal
    frames of the cohort's own goal store reads the cube within B_cube = 2.397 cm in median.
    Reported beside it: the per-subgoal medians and the median on each goal demonstration's
    first frame (the reset frame, re-rendered from the declared reset)."""
    from embodied_jepa import so101_wm as sw

    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    require_v1(goals, cl.LEDGER_COHORTS)
    report["goals_sha256"] = goals["sha256"]
    report["cohort"] = goals["report"]["cohort"]
    shared = rt.shared_root()
    rr = json.loads((shared / cl.TASK100_READOUTS / "report.json").read_text())
    path = shared / cl.TASK100_READOUTS / cl.RENC_FILE
    if cl.sha256_file(path) != rr["files_sha256"][cl.RENC_FILE]:
        raise cl.GuardError("G-readouts: renc_pretrained.npz differs from TASK-100's report")
    fits = sw.load_readouts(path)
    moments = rt.load_moments()
    a = goals["arrays"]
    x = rt.standardise(moments, a["latents_raw"].reshape(-1, cl.TOKENS, cl.WIDTH))
    pred = sw.apply_readout(fits["cube"], x)
    err = sw.label_errors("cube", pred, a["cube_cm"].reshape(-1, 3))
    median = float(np.median(err))
    per_subgoal = {str(k + 1): float(np.median(err.reshape(-1, 4)[:, k])) for k in range(4)}
    bar = sw.frozen_choices()["bars"]["B_cube"]
    feat = rt.LiveFeaturiser(args.device)
    reset_err = []
    for seed in [int(x) for x in a["seeds"]]:
        robot = rt.make_robot(True)
        try:
            rt.lift_task(robot).reset(seed)
            frames = {c: robot.sim.render(c)[None] for c in cl.CAMERAS}
            truth = robot.sim.data.body("cube").xpos.copy() * 100.0
        finally:
            robot.close()
        _, z = feat.encode(frames)
        pred_r = sw.apply_readout(fits["cube"], z)
        reset_err.append(float(sw.label_errors("cube", pred_r, truth[None])[0]))
    return {
        "frames": int(len(err)),
        "median_cm": median,
        "B_cube": bar,
        "per_subgoal_median_cm": per_subgoal,
        "reset_frame_median_cm": float(np.median(reset_err)),
        "reset_frames": len(reset_err),
        "passes": bool(median <= bar),
    }


def check_projector(args, report) -> dict:
    """§7.2.3 (A1: on K′'s reset states): 1 000 random 16-step sequences from the goal store's
    reset states: the vectorised chained projection equals ``preview`` chained step by step and
    the applied actions ``execute`` returns."""
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    require_v1(goals, ("Kp",))
    report["goals_sha256"] = goals["sha256"]
    report["cohort"] = goals["report"]["cohort"]
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    rng = np.random.default_rng([cl.SALTS["rand"], 7])  # plumbing draws for the check only
    preview_equal = execute_equal = stopped = compared = 0
    robot = rt.make_robot(False)
    task = rt.lift_task(robot)
    params = cl.PreviewParams.of(robot)
    try:
        for i in range(args.sequences):
            task.reset(seeds[i % len(seeds)])
            req = rng.uniform(-1, 1, (16, 6)).astype(np.float32)
            if i % 4 == 0:  # also exercise saturated commands (the joint limits)
                req = np.sign(req).astype(np.float32)
            vec, _ = cl.project_sequences(req[None], robot.sim.targets.copy(), params)
            t = robot.sim.targets.copy()
            chained = []
            for step in req:
                t, applied, _ = robot.preview(step, t)
                chained.append(applied)
            preview_equal += int(np.array_equal(vec[0], np.asarray(chained)))
            executed, ok = [], True
            for step in req:
                robot.observe()
                r = robot.execute(step)
                if r.status == "stopped":
                    ok = False
                    break
                executed.append(r.applied_action)
            if ok:
                compared += 1
                execute_equal += int(np.array_equal(vec[0], np.asarray(executed)))
            else:
                stopped += 1
    finally:
        robot.close()
    n = args.sequences
    return {
        "sequences": n,
        "preview_equal": preview_equal,
        "execute_compared": compared,
        "execute_equal": execute_equal,
        "execute_stopped": stopped,
        "passes": preview_equal == n and execute_equal == compared and compared > 0,
    }


def check_ties(args, report) -> dict:
    """§6/§7.1: N_J-100101's costs are bit-identical across the batch on CUDA, and the planner
    returns its initial mean (the zero command, projected)."""
    from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
    from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner

    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    feat = rt.LiveFeaturiser(args.device)
    model, _ = rt.load_model(cl.TWIN, args.device)
    z = feat.standardise(goals["arrays"]["latents_raw"][0, :1])[0]
    g = feat.standardise(goals["arrays"]["latents_raw"][0, 3:4])[0]
    out = {}
    robot = rt.make_robot(False)
    params = cl.PreviewParams.of(robot)
    targets = robot.sim.targets.copy()
    robot.close()
    for cost in cl.COSTS:
        for h in cl.HORIZONS:
            adapter = rt.LatentCost(model, cost, action_blind=True)
            rng = np.random.default_rng(h)
            cand = rng.uniform(-1, 1, (1, cl.CEM["samples"], h, 6)).astype(np.float32)
            d = adapter.distance(adapter.predict(z, cand), g)
            planner = SchemaCEMPlanner(
                SchemaCEMConfig(
                    action_schema=JOINT_DELTA_SO101_V0,
                    horizon=h,
                    samples=cl.CEM["samples"],
                    iterations=cl.CEM["iterations"],
                    elites=cl.CEM["elites"],
                    seed=cl.planner_seed(0),
                    minimum_std=cl.CEM["minimum_std"],
                    project_candidates=True,
                )
            )
            plan = planner.plan(adapter, z, g, projector=rt.Projector(params, targets))
            zero_proj, _ = cl.project_sequences(np.zeros((1, h, 6), np.float32), targets, params)
            # the non-broadcast batched roll-out of identical inputs, for the record
            batched = model.predict_features(
                np.repeat(z[None], 256, 0), np.zeros((256, h, 6), np.float32)
            )
            out[f"H{h}-{cost}"] = {
                "costs_bit_identical": bool((d == d[:, :1]).all()),
                "requested_is_initial_mean": bool(
                    np.array_equal(plan.requested_actions[0], np.zeros((h, 6), np.float32))
                ),
                "executed_is_projected_zero": bool(np.array_equal(plan.actions[0], zero_proj[0])),
                "unbroadcast_rows_bit_identical": bool((batched == batched[:1]).all()),
            }
    out["passes"] = all(
        v["costs_bit_identical"]
        and v["requested_is_initial_mean"]
        and v["executed_is_projected_zero"]
        for v in out.values()
    )
    return out


def check_adapter(args, report) -> dict:
    """The adapter's per-step errors against TASK-100's ``predict_features`` + ``latent_error``
    on the same candidates (CUDA; reported)."""
    from embodied_jepa import so101_wm as sw

    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    feat = rt.LiveFeaturiser(args.device)
    model, _ = rt.load_model(cl.PRIMARY, args.device)
    raw = goals["arrays"]["latents_raw"]
    out = {}
    for h in cl.HORIZONS:
        z = feat.standardise(raw[0, :1])[0]
        g = feat.standardise(raw[0, 1:2])[0]
        cand = np.random.default_rng(h).uniform(-1, 1, (1, 256, h, 6)).astype(np.float32)
        adapter = rt.LatentCost(model, "last")
        mine = adapter.step_errors(adapter.predict(z, cand), g)
        pred = model.predict_features(np.repeat(z[None], 256, 0), cand[0])
        ref = sw.latent_error(pred.reshape(-1, cl.LATENT_DIM), np.repeat(g[None], 256 * h, 0))
        ref = ref.reshape(256, h)
        out[f"H{h}"] = {
            "max_abs": float(np.abs(mine - ref).max()),
            "max_rel": float((np.abs(mine - ref) / ref).max()),
            "argsort_equal_last": bool(
                np.array_equal(
                    np.argsort(mine[:, -1], kind="stable"), np.argsort(ref[:, -1], kind="stable")
                )
            ),
        }
    out["passes"] = all(v["max_rel"] < 1e-5 for v in out.values())
    return out


ACTION_KEYS = ("applied", "requested", "status", "subgoal")


def _first_difference(x, y) -> int | None:
    """First index at which two per-step arrays differ (a length difference counts there)."""
    n = min(len(x), len(y))
    a = np.asarray(x[:n]).reshape(n, -1)
    b = np.asarray(y[:n]).reshape(n, -1)
    same = np.all((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b, 1)
    bad = np.flatnonzero(~same)
    if len(bad):
        return int(bad[0])
    return None if len(x) == len(y) else n


def pair_divergence(path_a: Path, path_b: Path, ra: dict, rb: dict) -> dict:
    """Where two runs of one episode first differ: in the live latent (the rendered frames) and
    in the executed actions; the outcome keys."""
    with np.load(path_a) as x, np.load(path_b) as y:
        latent = _first_difference(x["latents_raw16"], y["latents_raw16"])
        actions = [_first_difference(x[k], y[k]) for k in ACTION_KEYS]
        unstable = int(
            sum(
                not np.array_equal(u, v)
                for u, v in zip(x["latents_raw16"], y["latents_raw16"], strict=False)
            )
        )
    acts = [a for a in actions if a is not None]
    action = min(acts) if acts else None
    outcome_keys = ("success", "termination", "success_step", "commands")
    same_outcome = all(ra[k] == rb[k] for k in outcome_keys)
    identical = latent is None and action is None and same_outcome
    # the controller and the physics are deterministic: actions never diverge before the frames
    # do, and with identical frames throughout, the outcome is identical
    conditional = (action is None and same_outcome) or (
        latent is not None and action is not None and action >= latent
    )
    return {
        "first_latent_difference": latent,
        "first_action_difference": action,
        "steps_with_different_latents": unstable,
        "same_outcome": same_outcome,
        "bit_identical": identical,
        "conditional_determinism": bool(conditional),
    }


def check_determinism(args, report) -> dict:
    """§7.2.4: one W and one N episode run twice; four W episodes in a pool of 1 and of
    ``--workers``.

    As declared, the executed sequences and outcomes must be identical ("literal"). MuJoCo's EGL
    renderer returns different images for the same physics state on some frames (one pixel, one
    intensity level; Stage 0 record), so the literal form cannot hold whenever such a frame
    occurs. The check passes on the conditional form (Stage 0 record's deviation): in every pair
    the executed actions never diverge before the live latent (the rendered frames) does, and
    with identical latents at every step the actions and the outcome are identical. Both forms
    are reported."""
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    require_v1(goals, ("debug",))
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    base = Path(args.output)
    cfg = args.config
    runs = {}

    def run(tag, arm, slots, workers):
        folder = base / tag
        folder.mkdir(parents=True)
        specs = [
            {
                "arm": arm,
                "seed": seeds[s],
                "slot": s,
                "config": cfg,
                "keep_frames": False,
                "output": str(folder),
            }
            for s in slots
        ]
        t0 = time.monotonic()
        res = rt.run_pool(
            specs,
            goals_dir=args.goals,
            goals_sha256=goals["sha256"],
            device=args.device,
            workers=workers,
        )
        runs[tag] = {"results": res, "wall_seconds": time.monotonic() - t0}
        return {r["seed"]: r for r in res}

    w1 = run("W-a", "W", [0], 1)
    w2 = run("W-b", "W", [0], 1)
    n1 = run("N-a", "N", [0], 1)
    n2 = run("N-b", "N", [0], 1)
    p1 = run("pool-1", "W", [0, 1, 2, 3], 1)
    pk = run(f"pool-{args.workers}", "W", [0, 1, 2, 3], args.workers)
    s0 = seeds[0]

    def pair(x, y, tag_x, tag_y, seed):
        return pair_divergence(
            base / tag_x / f"{seed}.npz", base / tag_y / f"{seed}.npz", x[seed], y[seed]
        )

    pk_tag = f"pool-{args.workers}"
    pairs = {
        "W_rerun": pair(w1, w2, "W-a", "W-b", s0),
        "N_rerun": pair(n1, n2, "N-a", "N-b", s0),
        **{f"pool_{s}": pair(p1, pk, "pool-1", pk_tag, s) for s in seeds[:4]},
    }
    out = {
        "pairs": pairs,
        "literal_identical": all(v["bit_identical"] for v in pairs.values()),
        "conditional_determinism": all(v["conditional_determinism"] for v in pairs.values()),
        "workers": args.workers,
        "runs": {
            k: {
                "wall_seconds": v["wall_seconds"],
                "episodes": [
                    {
                        x: r[x]
                        for x in (
                            "seed",
                            "termination",
                            "commands",
                            "plan_seconds_median",
                            "plan_seconds_p95",
                            "episode_seconds",
                        )
                    }
                    for r in v["results"]
                ],
            }
            for k, v in runs.items()
        },
    }
    out["passes"] = out["conditional_determinism"]
    return out


def check_render(args, report) -> dict:
    """Reported (Stage 0 record): how often MuJoCo's EGL renderer returns different images for
    one physics state. Follower v1 on the first ``--render-seeds`` debug resets; after every
    command both cameras are rendered ``--render-repeats`` times."""
    from embodied_jepa.so101_followers import follow_lift

    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    if goals["report"]["cohort"] != "debug":
        raise cl.GuardError("the render check runs on debug seeds")
    rows = []
    for seed in [int(x) for x in goals["arrays"]["seeds"][: args.render_seeds]]:
        robot = rt.make_robot(True)
        task = rt.lift_task(robot)
        counts = {"states": 0, "unstable": {c: 0 for c in cl.CAMERAS}, "max_pixels": 0}

        def on_command(n, result, robot=robot, counts=counts):
            counts["states"] += 1
            for c in cl.CAMERAS:
                images = [robot.sim.render(c) for _ in range(args.render_repeats)]
                if any(not np.array_equal(images[0], x) for x in images[1:]):
                    counts["unstable"][c] += 1
                    px = max(
                        int((np.abs(images[0].astype(int) - x).max(-1) > 0).sum())
                        for x in images[1:]
                    )
                    counts["max_pixels"] = max(counts["max_pixels"], px)

        try:
            follow_lift(rt.Recorder(robot, on_command), task, seed)
        finally:
            robot.close()
        rows.append({"seed": seed} | counts)
    for path in sorted(Path(args.replay).glob("*.npz")) if args.replay else []:
        with np.load(path) as log:
            requested = log["requested"]
        seed = int(path.stem)
        robot = rt.make_robot(True)
        task = rt.lift_task(robot)
        counts = {"states": 0, "unstable": {c: 0 for c in cl.CAMERAS}, "max_pixels": 0}
        try:
            task.reset(seed)
            for action in requested:
                robot.observe()
                if robot.execute(action).status == "stopped":
                    break
                counts["states"] += 1
                for c in cl.CAMERAS:
                    images = [robot.sim.render(c) for _ in range(args.render_repeats)]
                    if any(not np.array_equal(images[0], x) for x in images[1:]):
                        counts["unstable"][c] += 1
                        px = max(
                            int((np.abs(images[0].astype(int) - x).max(-1) > 0).sum())
                            for x in images[1:]
                        )
                        counts["max_pixels"] = max(counts["max_pixels"], px)
        finally:
            robot.close()
        rows.append({"seed": seed, "replay": str(path)} | counts)
    states = sum(r["states"] for r in rows)
    unstable = {c: sum(r["unstable"][c] for r in rows) for c in cl.CAMERAS}
    return {
        "rows": rows,
        "states": states,
        "unstable": unstable,
        "repeats": args.render_repeats,
        "passes": True,
    }


def check_marker(args, report) -> dict:
    """Reported diagnostic of a §7.2.2 miss: D's goal demonstrations replayed, each subgoal state
    rendered twice -- as the task declares it (the lift scene's target marker at x 0.30, y -0.20 m)
    and with only the marker (no contact, ``contype`` 0) moved to a point drawn from the play
    corpus's marker box (x 0.16-0.22, |y| <= 0.14, >= 8 cm from the cube; rng 0) -- and R-enc's
    cube error read on both. The marker is restored before the next physics step. (Stage 0's
    diagnostic on D's v0 goal store; it runs ``so101_cube_lift_v0`` and D is spent under A1.)"""
    from embodied_jepa import so101_wm as sw
    from embodied_jepa.so101_followers import follow_lift
    from embodied_jepa.so101_tasks import LIFT_TASK, SO101CubeLiftTask

    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    if goals["report"]["cohort"] != "D":
        raise cl.GuardError("the marker diagnostic reads D's goal demonstrations")
    shared = rt.shared_root()
    rr = json.loads((shared / cl.TASK100_READOUTS / "report.json").read_text())
    path = shared / cl.TASK100_READOUTS / cl.RENC_FILE
    if cl.sha256_file(path) != rr["files_sha256"][cl.RENC_FILE]:
        raise cl.GuardError("G-readouts: renc_pretrained.npz differs from TASK-100's report")
    fits = sw.load_readouts(path)
    feat = rt.LiveFeaturiser(args.device)
    rng = np.random.default_rng(0)
    errors = {"declared": [], "marker_in_corpus_box": []}
    stored = goals["arrays"]["latents_raw"].reshape(-1, cl.TOKENS, cl.WIDTH)
    rerendered = []
    for seed in [int(s) for s in goals["arrays"]["seeds"]]:
        robot = rt.make_robot(True)
        task = SO101CubeLiftTask(robot)
        assert task.name == LIFT_TASK
        sim = robot.sim
        cube_xy = np.array(task.sample_reset(seed)["cube_xy"])
        while True:
            marker = np.array([rng.uniform(0.16, 0.22), rng.uniform(-0.14, 0.14)])
            if np.linalg.norm(marker - cube_xy) >= 0.08:
                break
        caps = []

        def on_command(n, result, sim=sim, marker=marker, caps=caps):
            if n not in cl.SUBGOAL_COMMANDS:
                return
            declared = {c: sim.render(c) for c in cl.CAMERAS}
            old = sim.model.body("target").pos.copy()
            sim.model.body("target").pos[:2] = marker
            sim.mj.mj_kinematics(sim.model, sim.data)
            moved = {c: sim.render(c) for c in cl.CAMERAS}
            sim.model.body("target").pos[:] = old
            sim.mj.mj_kinematics(sim.model, sim.data)
            caps.append((declared, moved, sim.data.body("cube").xpos.copy() * 100.0))

        try:
            follow_lift(rt.Recorder(robot, on_command), task, seed)
        finally:
            robot.close()
        for declared, moved, truth in caps:
            for key, frames in (("declared", declared), ("marker_in_corpus_box", moved)):
                raw, z = feat.encode({c: frames[c][None] for c in cl.CAMERAS})
                if key == "declared":
                    rerendered.append(raw[0])
                pred = sw.apply_readout(fits["cube"], z)
                errors[key].append(float(sw.label_errors("cube", pred, truth[None])[0]))
    same = int(sum(np.array_equal(a, b) for a, b in zip(rerendered, stored, strict=True)))
    bar = sw.frozen_choices()["bars"]["B_cube"]
    return {
        "frames": len(errors["declared"]),
        "median_cm": {k: float(np.median(v)) for k, v in errors.items()},
        "B_cube": bar,
        "declared_latents_equal_goal_store": same,
        "passes": None,
    }


# ----- A1's new checks (§18.5) --
def _sha(rel: str) -> str:
    import hashlib

    return hashlib.sha256((rt.ROOT / rel).read_bytes()).hexdigest()


def check_scene(args, report) -> dict:
    """§7.2.0 (A1): every declared scene element of ``so101_cube_lift_v1`` against the
    ``so101-play-v1`` corpus's scene distribution, filled in from the code; the simulation, MJCF
    and action configuration against their TASK-099 pins; the cube and marker of every K', D', S
    and X seed (main and reserve) drawn without a simulator. Reads no frame and no count."""
    from embodied_jepa import so101_play_corpus as play
    from embodied_jepa import so101_tasks as tasks
    from embodied_jepa import so101_tasks_lift_v1 as v1

    play_pins = json.loads((rt.ROOT / "benchmarks/manifests/so101-play-v1.json").read_text())
    action = json.loads((rt.ROOT / "configs/so101_sim_action.json").read_text())
    assets = json.loads((rt.ROOT / "assets/so101_manifest.json").read_text())
    home = np.asarray(action["home_rad"], float)
    d = tasks.RESETS_V0

    def sub(inner, outer):
        return bool(outer[0] <= inner[0] <= inner[1] <= outer[1])

    rows = []

    def row(element, corpus, lift, inside):
        rows.append({"element": element, "corpus": corpus, "lift_v1": lift, "inside": bool(inside)})

    row(
        "target marker xy",
        {"box": play.TARGET_REGION, "min_cube_m": play.MIN_CUBE_TARGET_M},
        {"box": v1.MARKER_REGION, "min_cube_m": v1.MIN_CUBE_MARKER_M, "salt": v1.MARKER_SALT},
        v1.MARKER_REGION is play.TARGET_REGION and v1.MIN_CUBE_MARKER_M == play.MIN_CUBE_TARGET_M,
    )
    row(
        "cube xy",
        {"x": play.CUBE_REGION[0], "y": play.CUBE_REGION[1]},
        {"x": d.cube_x_m, "y": d.cube_y_m},
        sub(d.cube_x_m, play.CUBE_REGION[0]) and sub(d.cube_y_m, play.CUBE_REGION[1]),
    )
    corpus_yaw = (-np.pi / 4, np.pi / 4)  # so101_play_corpus.sample_reset's literal
    row("cube yaw", {"rad": corpus_yaw}, {"rad": d.cube_yaw_rad}, sub(d.cube_yaw_rad, corpus_yaw))
    arm_box = np.stack([home[:5] - play.HOME_JITTER_RAD, home[:5] + play.HOME_JITTER_RAD], 1)
    row(
        "arm at reset",
        {"home_rad": home[:5], "jitter_rad": play.HOME_JITTER_RAD},
        {"rad": home[:5]},
        bool(np.all((arm_box[:, 0] <= home[:5]) & (home[:5] <= arm_box[:, 1]))),
    )
    row(
        "jaw at reset",
        {"rad": play.HOME_GRIPPER_RAD},
        {"rad": float(home[5])},
        play.HOME_GRIPPER_RAD[0] <= home[5] <= play.HOME_GRIPPER_RAD[1],
    )
    pinned = {}
    for rel in (
        "src/embodied_jepa/so101_simulation.py",
        "configs/so101_sim_action.json",
        "assets/so101_manifest.json",
    ):
        pinned[rel] = {"sha256": _sha(rel), "pin": play_pins["code_sha256"][rel]}
    mjcf = assets["robot"]
    mjcf_path = Path(rt.ROOT / assets["local_path"]) / mjcf
    import hashlib

    pinned[f"{assets['local_path']}/{mjcf}"] = {
        "sha256": hashlib.sha256(mjcf_path.read_bytes()).hexdigest(),
        "pin": assets["sha256"][mjcf],
    }
    files_ok = all(v["sha256"] == v["pin"] for v in pinned.values())
    row(
        "table, cube size and colour, marker size and colour, lighting",
        {"so101_simulation.py": pinned["src/embodied_jepa/so101_simulation.py"]["pin"]},
        {"so101_simulation.py": pinned["src/embodied_jepa/so101_simulation.py"]["sha256"]},
        pinned["src/embodied_jepa/so101_simulation.py"]["sha256"]
        == pinned["src/embodied_jepa/so101_simulation.py"]["pin"],
    )
    row(
        "cameras, 224 x 224, renderer",
        {"cameras": play_pins["cameras"], "image_size": play_pins["image_size"]},
        {"cameras": list(cl.CAMERAS), "image_size": cl.IMAGE_SIZE, "files": pinned},
        files_ok
        and list(play_pins["cameras"]) == list(cl.CAMERAS)
        and int(np.ravel(play_pins["image_size"])[0]) == cl.IMAGE_SIZE,
    )
    # every seed the amended study can draw: cubes equal v0's, markers inside the box and >= 8 cm
    draws = {"seeds": 0, "cube_equals_v0": 0, "marker_in_box": 0, "marker_far": 0, "cube_in": 0}
    for name in ("Kp", "Dp", "S", "X"):
        c = cl.COHORTS[name]
        for seed in (*c.seeds, *c.reserve_seeds):
            spec = v1.sample_reset_lift_v1(seed)
            v0 = tasks.sample_reset(tasks.LIFT_TASK, seed)
            cube, marker = np.asarray(spec["cube_xy"]), np.asarray(spec["target_xy"])
            draws["seeds"] += 1
            draws["cube_equals_v0"] += int(
                spec["cube_xy"] == v0["cube_xy"] and spec["cube_yaw"] == v0["cube_yaw"]
            )
            draws["marker_in_box"] += int(
                play.TARGET_REGION[0][0] <= marker[0] <= play.TARGET_REGION[0][1]
                and play.TARGET_REGION[1][0] <= marker[1] <= play.TARGET_REGION[1][1]
            )
            draws["marker_far"] += int(np.linalg.norm(marker - cube) >= play.MIN_CUBE_TARGET_M)
            draws["cube_in"] += int(
                play.CUBE_REGION[0][0] <= cube[0] <= play.CUBE_REGION[0][1]
                and play.CUBE_REGION[1][0] <= cube[1] <= play.CUBE_REGION[1][1]
            )
    row(
        "drawn resets (K', D', S, X with reserves)",
        "inside the corpus's support",
        draws,
        all(v == draws["seeds"] for v in draws.values()),
    )
    rows.append(
        {
            "element": "motion",
            "corpus": "scripted play so101_play_policy_v0",
            "lift_v1": "follower v1 demonstrations; CEM's own commands",
            "inside": None,
            "note": "not compared by construction (§18.4); §14's caveat stands",
        }
    )
    return {
        "rows": rows,
        "files": pinned,
        "task": v1.LIFT_V1_TASK,
        "passes": all(r["inside"] for r in rows if r["inside"] is not None),
    }


INERT_TRUTH = (
    "cube_position",
    "cube_quaternion",
    "cube_velocity",
    "cube_angular_velocity",
    "gripper_site_position",
    "fixed_jaw_contact",
    "moving_jaw_contact",
    "gripper_contact",
    "timestamp",
)


def _inert_trace(seed: int, variant: str) -> dict:
    """The goal demonstration on ``variant``'s reset, rendering off; after the reset and after
    every command: ``qpos``, ``qvel``, the transport's targets, the executed commands and the
    named ``task_truth()`` arrays, plus the marker position."""
    from embodied_jepa.so101_followers import follow_lift

    robot = rt.make_robot(False)
    task = rt.lift_task(robot, variant)
    trace: dict = {k: [] for k in ("qpos", "qvel", "targets", "target_position", *INERT_TRUTH)}
    commands: list = []

    def snap():
        sim = robot.sim
        truth = sim.task_truth()
        trace["qpos"].append(sim.data.qpos[sim.qadr].copy())
        trace["qvel"].append(sim.data.qvel[sim.vadr].copy())
        trace["targets"].append(sim.targets.copy())
        trace["target_position"].append(np.asarray(truth["target_position"]).copy())
        for k in INERT_TRUTH:
            trace[k].append(np.asarray(truth[k]).copy())

    class Tap:
        def __getattr__(self, name):
            return getattr(robot, name)

        def execute(self, action):
            if not commands:
                snap()  # the state after the reset
            result = robot.execute(action)
            commands.append(
                np.asarray(action, np.float32).copy()
                if result.applied_action is None
                else np.asarray(result.applied_action, np.float32).copy()
            )
            snap()
            return result

    try:
        record = follow_lift(Tap(), task, seed)
    finally:
        robot.close()
    out = {k: np.asarray(v) for k, v in trace.items()}
    out["commands"] = np.asarray(commands, np.float32)
    out["target_xy"] = np.asarray(task.reset_spec["target_xy"], float)
    out["success"] = bool(record["success"])
    return out


def check_inert(args, report) -> dict:
    """§7.2.6 (A1): the marker is inert. On debug seeds 101900-101915 the goal demonstration runs
    with v0's and with v1's reset, rendering off; every named array must be bit-identical between
    them after the reset and after every command, and the marker position must equal
    ``(target_xy, TABLE_TOP_Z)`` exactly at every step (the drawn point in v1, (0.30, -0.20) in
    v0). Reads no frame and no count."""
    from embodied_jepa.so101_tasks import LIFT_TASK, TABLE_TOP_Z
    from embodied_jepa.so101_tasks_lift_v1 import LIFT_V1_TASK, V0_MARKER_XY

    rows = []
    for seed in range(cl.COHORTS["debug"].first, cl.COHORTS["debug"].first + args.inert_seeds):
        cl.check_seed("debug", seed)
        a, b = _inert_trace(seed, LIFT_TASK), _inert_trace(seed, LIFT_V1_TASK)
        keys = ("qpos", "qvel", "targets", "commands", *INERT_TRUTH)
        differing = [k for k in keys if not np.array_equal(a[k], b[k])]
        marker_ok = {}
        for name, tr, want in (("v0", a, np.asarray(V0_MARKER_XY)), ("v1", b, b["target_xy"])):
            if name == "v0" and not np.array_equal(tr["target_xy"], want):
                marker_ok[name] = False
                continue
            expected = np.array([*tr["target_xy"], TABLE_TOP_Z])
            marker_ok[name] = bool(np.all(tr["target_position"] == expected))
        rows.append(
            {
                "seed": seed,
                "steps": len(a["qpos"]),
                "commands": len(a["commands"]),
                "differing": differing,
                "marker_exact": marker_ok,
                "v1_marker_xy": b["target_xy"].tolist(),
                "success": [a["success"], b["success"]],
            }
        )
    return {
        "seeds": len(rows),
        "rows": rows,
        "passes": bool(
            rows and all(not r["differing"] and all(r["marker_exact"].values()) for r in rows)
        ),
    }


# ----- the S0-NULL diagnostic Delta (§18.6) --
def cmd_delta(args) -> int:
    """Delta on cohort X's goal store, at the configuration K′'s rule chose (H8-mean when all
    tie at 0). One GPU job, one process; writes ``delta.json`` and the per-state rows."""
    out = Path(args.output)
    if out.exists():
        raise cl.GuardError(f"G-twice: {out} exists")
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    if goals["report"]["cohort"] != "X":
        raise cl.GuardError("Delta reads cohort X's goal store")
    if goals["report"].get("task_name") != cl.LIFT_TASK_NAME:
        raise cl.GuardError(f"G-goals: the goal store was not made on {cl.LIFT_TASK_NAME}")
    cl.parse_config(args.config)
    out.mkdir(parents=True)
    t0 = time.monotonic()
    report = rt.header("delta", config=args.config, goals_sha256=goals["sha256"])
    guards(report, False)
    gpu_prelude(report, args.device)
    featuriser = rt.LiveFeaturiser(args.device)
    ctx = rt.goal_context(goals, featuriser) | {"featuriser": featuriser}
    models: dict = {}

    def adapter(name, cost):
        if name not in models:
            models[name] = rt.load_model(name, args.device)[0]
        return rt.LatentCost(models[name], cost)

    ctx["adapter"] = adapter
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    rows = []
    for slot, seed in enumerate(seeds):
        r = rt.delta_seed(seed, slot, goals, ctx, args.config)
        rows.extend(r["states"])
        rt.log(f"delta {seed}: {[round(x['rank'], 3) for x in r['states']]}")
    h, _ = cl.parse_config(args.config)
    measures = cl.delta_measures(rows, h)
    report |= {
        "kept": len(seeds),
        "drawn": goals["report"]["drawn"],
        "rejected": goals["report"]["rejected"],
        "measures": measures,
        "states": rows,
        "seconds": time.monotonic() - t0,
        "finished_utc": rt.utc(),
    }
    rt.write_json(out / "delta.json", report)
    rt.log(f"delta: m1 {measures['all']['m1']}, m2 {measures['all']['m2']}")
    return 0


CHECKS = {
    "latent": check_latent,
    "renc": check_renc,
    "projector": check_projector,
    "ties": check_ties,
    "adapter": check_adapter,
    "determinism": check_determinism,
    "render": check_render,
    "marker": check_marker,
    "scene": check_scene,
    "inert": check_inert,
}


def cmd_check(args) -> int:
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    report = rt.header("check", check=args.check, device=args.device)
    guards(report, False)
    if args.device == "cuda":
        gpu_prelude(report, args.device)
    t0 = time.monotonic()
    report["result"] = CHECKS[args.check](args, report)
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = rt.utc()
    sha = rt.write_json(out / "report.json", report)
    # §18.5 (2): every check writes a ledger entry into the run's output directory
    cl.ledger_append(
        ledger_path(args),
        cl.ledger_entry(
            check=args.check,
            passes=report["result"].get("passes"),
            revision=report["revision"],
            dirty=report["tracked_tree_dirty"],
            finished_utc=report["finished_utc"],
            goals_sha256=report.get("goals_sha256"),
            cohort_name=report.get("cohort"),
            report_sha256=sha,
        )
        | {"report": str(out / "report.json")},
    )
    rt.log(f"check {args.check}: passes={report['result'].get('passes')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=sorted(cl.COHORTS))
    g.add_argument("--output", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--gated", action="store_true")
    e = sub.add_parser("episodes")
    e.add_argument("--cohort", required=True, choices=sorted(cl.COHORTS))
    e.add_argument("--goals", required=True)
    e.add_argument("--goals-sha256", required=True)
    e.add_argument("--arms", required=True)
    e.add_argument("--configs", default=None)
    e.add_argument("--slots", default=None)
    e.add_argument("--evidence", type=int, default=cl.EVIDENCE_FRAMES_EPISODES)
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--device", default="cuda")
    e.add_argument("--output", required=True)
    e.add_argument("--gated", action="store_true")
    e.add_argument("--ledger", default=None)
    c = sub.add_parser("check")
    c.add_argument("check", choices=sorted(CHECKS))
    c.add_argument("--goals", default=None)
    c.add_argument("--goals-sha256", default=None)
    c.add_argument("--device", default="cuda")
    c.add_argument("--sequences", type=int, default=1000)
    c.add_argument("--config", default="H8-mean")
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--render-seeds", type=int, default=4)
    c.add_argument("--render-repeats", type=int, default=3)
    c.add_argument("--replay", default=None)
    c.add_argument("--inert-seeds", type=int, default=16)
    c.add_argument("--ledger", default=None)
    c.add_argument("--output", required=True)
    x = sub.add_parser("delta")
    x.add_argument("--goals", required=True)
    x.add_argument("--goals-sha256", required=True)
    x.add_argument("--config", default="H8-mean")
    x.add_argument("--device", default="cuda")
    x.add_argument("--output", required=True)
    args = p.parse_args(argv)
    return {"goals": cmd_goals, "episodes": cmd_episodes, "check": cmd_check, "delta": cmd_delta}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
