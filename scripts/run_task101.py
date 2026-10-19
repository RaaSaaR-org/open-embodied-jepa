"""TASK-101's runner (protocol docs/experiments/so101_closed_loop_lift_v1.md §7, §8, §12).

Subcommands (each refuses an existing output):

* ``goals``    -- the cohort's goal demonstrations (follower v1, rendered), §4's replacement
  from the reserve, the subgoal frames encoded by the live latent, the hashed goal store; with
  ``--evidence N`` also the first N episodes of the CPU arms (rand, hold, F, H-pose) rendered for
  the evidence (GPU job: EGL rendering and DINOv2).
* ``episodes`` -- arms on a cohort's kept resets, one pool of workers (GPU for CEM arms; CPU arms
  run with ``--device none`` outside the lock, without frames).
* ``check``    -- §7.2's checks: ``latent`` (live = stored), ``renc`` (the live latent reads the
  cube), ``projector`` (chained preview = execute), ``ties`` (N's costs, CUDA), ``adapter`` (the
  adapter against ``predict_features``) and ``determinism`` (reruns and pool sizes).

Stage 0 runs on K, D (K', D' only under §7.6 row 1) and debug seeds; S refuses until the
manifest is FROZEN (``--gated``).
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


def check_cohort(name: str, gated: bool) -> None:
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
    report = rt.header("goals", cohort=args.cohort, evidence=args.evidence)
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
    if args.evidence:
        ev = out / "cpu-evidence"
        ev.mkdir()
        specs = []
        for arm in cl.CPU_ARMS:
            (ev / arm).mkdir()
            for slot in range(min(args.evidence, n)):
                specs.append(
                    {
                        "arm": arm,
                        "seed": seeds[slot],
                        "slot": slot,
                        "config": None,
                        "keep_frames": True,
                        "output": str(ev / arm),
                    }
                )
        results = rt.run_pool(specs, goals_dir=out, goals_sha256=sha, device="none", workers=1)
        rt.write_json(ev / "report.json", {"episodes": results, "seconds": time.monotonic() - t0})
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
    per call, as the controller does), equal to the stored float16 within 2 float16 ulps."""
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
        for f in np.linspace(0, n - 1, 8).astype(int):
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
                    "exact_fraction": float((diff == 0).mean()),
                }
            )
    return {"frames": count, "max_ulps": worst, "passes": bool(worst <= 2.0), "rows": rows}


def check_renc(args, report) -> dict:
    """§7.2.2: TASK-100's pretrained R-enc on the live-encoded subgoal frames of D's goal
    demonstrations reads the cube within B_cube = 2.397 cm in median."""
    from embodied_jepa import so101_wm as sw

    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    if goals["report"]["cohort"] != "D":
        raise cl.GuardError("the R-enc check reads D's goal demonstrations")
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
    return {
        "frames": int(len(err)),
        "median_cm": median,
        "B_cube": bar,
        "per_subgoal_median_cm": per_subgoal,
        "passes": bool(median <= bar),
    }


def check_projector(args, report) -> dict:
    """§7.2.3: 1 000 random 16-step sequences from D's first frames: the vectorised chained
    projection equals ``preview`` chained step by step and the applied actions ``execute``
    returns."""
    from embodied_jepa.so101_tasks import SO101CubeLiftTask

    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    rng = np.random.default_rng([cl.SALTS["rand"], 7])  # plumbing draws for the check only
    preview_equal = execute_equal = stopped = compared = 0
    robot = rt.make_robot(False)
    task = SO101CubeLiftTask(robot)
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


def compare_runs(a: Path, b: Path, keys=("applied", "requested", "status", "subgoal")) -> bool:
    with np.load(a) as x, np.load(b) as y:
        return all(np.array_equal(x[k], y[k], equal_nan=True) for k in keys)


def check_determinism(args, report) -> dict:
    """§7.2.4: one W and one N episode run twice; four W episodes in a pool of 1 and of
    ``--workers``; identical executed sequences and outcomes."""
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    if goals["report"]["cohort"] != "debug":
        raise cl.GuardError("the determinism check runs on debug seeds")
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

    def same(x, y, tag_x, tag_y, seed):
        files = compare_runs(base / tag_x / f"{seed}.npz", base / tag_y / f"{seed}.npz")
        keys = ("success", "termination", "success_step", "commands")
        return files and all(x[seed][k] == y[seed][k] for k in keys)

    out = {
        "W_rerun_identical": same(w1, w2, "W-a", "W-b", s0),
        "N_rerun_identical": same(n1, n2, "N-a", "N-b", s0),
        "pool_identical": all(same(p1, pk, "pool-1", f"pool-{args.workers}", s) for s in seeds[:4]),
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
    out["passes"] = out["W_rerun_identical"] and out["N_rerun_identical"] and out["pool_identical"]
    return out


CHECKS = {
    "latent": check_latent,
    "renc": check_renc,
    "projector": check_projector,
    "ties": check_ties,
    "adapter": check_adapter,
    "determinism": check_determinism,
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
    rt.write_json(out / "report.json", report)
    rt.log(f"check {args.check}: passes={report['result'].get('passes')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=sorted(cl.COHORTS))
    g.add_argument("--output", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--evidence", type=int, default=0)
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
    c = sub.add_parser("check")
    c.add_argument("check", choices=sorted(CHECKS))
    c.add_argument("--goals", default=None)
    c.add_argument("--goals-sha256", default=None)
    c.add_argument("--device", default="cuda")
    c.add_argument("--sequences", type=int, default=1000)
    c.add_argument("--config", default="H16-mean")
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--output", required=True)
    args = p.parse_args(argv)
    return {"goals": cmd_goals, "episodes": cmd_episodes, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
