"""TASK-102's runner (protocol docs/experiments/so101_hybrid_lift_v1.md §4, §6, §7, §8, §12).

Every job runs ``so101_cube_lift_v1`` (TASK-101's A1). Subcommands (each refuses an existing
output):

* ``goals``    -- a cohort's goal demonstrations (follower v1, rendered; TASK-101 §4's procedure
  and goal-store format), §4's replacement from the reserve, the subgoal frames encoded by the
  live latent, the hashed goal store.
* ``check``    -- §7.3's checks, each appended to the check ledger: ``latent`` (7.3.1, TASK-101's
  7.2.1 restated), ``renc`` (7.3.2, per cohort), ``projector`` (7.3.3, on P's reset states),
  ``gloc`` (7.3.4), ``determinism`` (7.3.5, conditional form; it also writes 7.3.6 ``timing``).
* ``episodes`` -- arms on a cohort's kept resets in one pool of workers. For P, A and D (and P′,
  A′, D′) it refuses, before any controller, simulator or seed is constructed, unless the check
  ledger admits it (TASK-101 §18.5's ordering rule). P runs S+G at the four declared offsets; A
  runs phase 1 alone (``H-approach``) in TASK-101's four configurations; D runs §6's arms with τ_h
  from P's file and the configuration from A's file.

Stage 0 runs on P, A, D and debug seeds; S refuses until the manifest is FROZEN (``--gated``).
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
from embodied_jepa import so101_hybrid as hy
from embodied_jepa import so101_hybrid_runtime as hr


def gpu_prelude(report: dict, device: str) -> None:
    from embodied_jepa import devices
    from embodied_jepa import run_tools as tools

    if device == "cuda":
        tools.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")


def guards(report: dict, gated: bool) -> None:
    if gated and hr.tree_dirty():
        raise hy.GuardError("G-tree: the tracked tree is dirty")
    hr.check_pins(report, gated=gated)


def ledger_path(args) -> Path:
    return Path(args.ledger) if args.ledger else Path(args.output).parent / hy.LEDGER_FILE


def check_cohort(name: str, gated: bool) -> None:
    hy.cohort(name)
    if name in hy.GATED_COHORTS and not gated:
        raise hy.GuardError("G-gated: the gated cohort runs only from the frozen revision")
    if name not in hy.GATED_COHORTS and gated:
        raise hy.GuardError("G-gated: --gated is for cohort S only")


def load_goals(args, cohorts=None) -> dict:
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    rep = goals["report"]
    if rep.get("task") != hy.TASK or rep.get("task_name") != hy.LIFT_TASK_NAME:
        raise hy.GuardError("G-goals: not a TASK-102 goal store on so101_cube_lift_v1")
    if cohorts is not None and rep["cohort"] not in cohorts:
        raise hy.GuardError(f"this job reads a goal store of {cohorts}")
    return goals


# ----- goals (§4) --
def cmd_goals(args) -> int:
    out = Path(args.output)
    check_cohort(args.cohort, args.gated)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    report = hr.header("goals", cohort=args.cohort, task_name=hy.LIFT_TASK_NAME)
    guards(report, args.gated)
    gpu_prelude(report, args.device)
    featuriser = rt.LiveFeaturiser(args.device)
    report["fit_report_sha256"] = featuriser.fit_report_sha256
    report["encoder_digest"] = featuriser.featuriser.digest
    demos: dict = {}

    def demonstrate(seed):
        hy.check_seed(args.cohort, seed)
        d = rt.demonstrate(seed, render=True)
        demos[seed] = d
        hr.log(f"demo {seed}: {d['termination_reason']} step {d['success_step']} kept={d['kept']}")
        return d["kept"], d["reason"], None

    if args.limit is not None:  # debug plumbing only
        if args.cohort != "debug":
            raise hy.GuardError("--limit is for the debug cohort only")
        hy.COHORTS["debug"] = hy.Cohort("debug", hy.COHORTS["debug"].first, args.limit, None)
    kept = hy.keep_cohort(args.cohort, demonstrate)
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
        caps = demos[s]["captured"]
        if [c["command"] for c in caps] != list(cl.SUBGOAL_COMMANDS):
            raise hy.GuardError(f"G-goals: {s}'s captures are not the four subgoals")
        for k, c in enumerate(caps):
            frames[i, k] = c["frames"]
            joints[i, k] = c["joints"]
            cube[i, k] = c["cube_cm"]
            site[i, k] = c["site"]
            xmat[i, k] = c["site_xmat"]
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
    hr.log(f"goal store {sha[:12]}: kept {n}, rejected {len(kept['rejected'])}")
    hr.write_json(
        out / "job.json",
        {"goals_sha256": sha, "seconds_total": time.monotonic() - t0, "finished_utc": hr.utc()},
    )
    return 0


# ----- episodes (§6, §7.4, §7.5) --
def parse_slots(text, n):
    if not text:
        return list(range(n))
    lo, hi = (int(x) for x in text.split(":"))
    return list(range(lo, min(hi, n)))


def stage_parameters(args) -> dict:
    """τ_h (from P's rule, ``--tau-file``) and phase 1's configuration (from A's rule,
    ``--config-file``); debug plumbing may name them directly."""
    out = {"tau_h": None, "config": None, "tau_file": None, "config_file": None}
    if args.tau_file:
        tau = hy.read_stage_parameter(args.tau_file)
        out["tau_h"], out["tau_file"] = (
            float(tau["tau_h"]),
            {
                "path": args.tau_file,
                "sha256": hr.sha256_file(args.tau_file),
            },
        )
    if args.config_file:
        cfg = hy.read_stage_parameter(args.config_file)
        out["config"], out["config_file"] = (
            cfg["chosen"],
            {
                "path": args.config_file,
                "sha256": hr.sha256_file(args.config_file),
            },
        )
    if args.tau_h is not None or args.config is not None:
        if args.cohort != "debug":
            raise hy.GuardError("--tau-h/--config are for debug plumbing; P and A set them")
        out["tau_h"] = args.tau_h if args.tau_h is not None else out["tau_h"]
        out["config"] = args.config if args.config is not None else out["config"]
    return out


def build_specs(args, seeds, params, out: Path) -> list[dict]:
    kind = hy.COHORT_KIND.get(args.cohort, "debug")
    arms = args.arms.split(",")
    for arm in arms:
        hy.check_arm(arm)
    if kind == "P" and arms != ["S+G"]:
        raise hy.GuardError("P runs S+G at the declared offsets only (§7.4)")
    if kind == "A" and arms != [hy.APPROACH_ARM]:
        raise hy.GuardError("A runs phase 1 alone (H-approach) only (§7.5)")
    if kind in ("D", "S") and not set(arms) <= set(hy.ARMS) - {hy.APPROACH_ARM}:
        raise hy.GuardError("D and S run §6's arms")
    configs = args.configs.split(",") if args.configs else [params["config"]]
    if kind == "A" and sorted(configs) != sorted(hy.CONFIGS):
        raise hy.GuardError("A runs every one of TASK-101's four configurations")
    if kind == "D" and (params["config"] is None or params["tau_h"] is None):
        raise hy.GuardError("D needs τ_h from P (--tau-file) and the configuration from A")
    if kind in ("A", "D") and params["tau_h"] is None:
        raise hy.GuardError("τ_h from P (--tau-file) is required")
    levels = [float(x) for x in args.levels.split(",")] if args.levels else [0.0]
    if kind == "P" and levels != list(hy.P_LEVELS_RAD):
        raise hy.GuardError(f"P runs every level {hy.P_LEVELS_RAD}")
    specs = []
    for slot in parse_slots(args.slots, len(seeds)):
        for arm in arms:
            p1 = hy.ARM_PLAN.get(arm, (None,))[0]
            cem = p1 is not None and p1.startswith("cem")
            if cem and args.device != "cuda":
                raise hy.GuardError("CEM arms run on the GPU")
            if arm in hy.GLOC_ARMS and args.device != "cuda":
                raise hy.GuardError("G_loc arms run on the GPU")
            for config in configs if cem else [None]:
                for level in levels if arm == "S+G" else [None]:
                    name = arm if config is None else f"{arm}@{config}"
                    if level is not None and kind == "P":
                        name = f"{arm}@r{level:.2f}"
                    folder = out / name
                    folder.mkdir(exist_ok=True)
                    specs.append(
                        {
                            "arm": arm,
                            "seed": seeds[slot],
                            "slot": slot,
                            "config": config,
                            "tau_h": params["tau_h"],
                            "level": level,
                            "keep_frames": slot < args.evidence,
                            "output": str(folder),
                        }
                    )
    return specs


def cmd_episodes(args) -> int:
    out = Path(args.output)
    check_cohort(args.cohort, args.gated)
    if out.exists():
        raise hy.GuardError(f"G-twice: {out} exists; a cohort and arm run once")
    goals = load_goals(args, (args.cohort,))
    verdict = None
    if args.cohort in hy.LEDGER_COHORTS:  # before any controller, simulator or seed
        if hr.tree_dirty():
            raise hy.GuardError("G-ledger: the tracked tree is dirty")
        verdict = hy.require_ledger(
            ledger_path(args), revision=hr.git("rev-parse", "HEAD"), goals_sha256=goals["sha256"]
        )
    params = stage_parameters(args)
    out.mkdir(parents=True)
    t0 = time.monotonic()
    report = hr.header(
        "episodes",
        cohort=args.cohort,
        arms=args.arms,
        configs=args.configs,
        levels=args.levels,
        workers=args.workers,
        goals_sha256=goals["sha256"],
        slots=args.slots,
        evidence=args.evidence,
        device=args.device,
        task_name=hy.LIFT_TASK_NAME,
        stage_parameters=params,
        gloc=None if not args.gloc else {"path": args.gloc, "sha256": hr.sha256_file(args.gloc)},
        ledger=None if verdict is None else str(ledger_path(args)),
        ledger_verdict=verdict,
    )
    guards(report, args.gated)
    if args.device == "cuda":
        gpu_prelude(report, args.device)
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    for s in seeds:
        hy.check_seed(args.cohort, s)
    specs = build_specs(args, seeds, params, out)
    report["episodes_planned"] = len(specs)
    progress = out / "progress.jsonl"
    done = []

    def on_result(r):
        done.append(r)
        with progress.open("a") as f:
            f.write(json.dumps(r, default=cl._json_default) + "\n")
        h = r.get("handover") or {}
        hr.log(
            f"{r['arm']}@{r.get('config')} r={r.get('level')} {r['seed']}: {r['termination']} "
            f"({r['commands']} cmds, hand-over {h.get('how')}@{h.get('command')}, "
            f"{r['episode_seconds']:.0f} s) [{len(done)}/{len(specs)}]"
        )

    report["episodes_started_utc"] = hr.utc()
    results = hr.run_pool(
        specs,
        goals_dir=args.goals,
        goals_sha256=goals["sha256"],
        device=args.device,
        workers=args.workers,
        gloc_file=args.gloc,
        on_result=on_result,
    )
    report["episodes"] = results
    report["wall_seconds"] = time.monotonic() - t0
    hr.check_pins(report, gated=args.gated)
    if hr.tree_dirty():
        raise hy.GuardError("G-tree: the tracked tree became dirty")
    report["revision_at_end"] = hr.git("rev-parse", "HEAD")
    report["finished_utc"] = hr.utc()
    hr.write_json(out / "report.json", report)
    return 0


# ----- checks (§7.3) --
def check_latent(args, report) -> dict:
    """7.3.1 = TASK-101's 7.2.1: 64 stored val frames per camera, encoded live one frame at a
    time, equal to the stored float16 within 2 ulps."""
    return hr.check_live_latent(args.device)


def check_renc(args, report) -> dict:
    """7.3.2 = TASK-101's 7.2.2 on this cohort's own goal store: TASK-100's pretrained R-enc on
    the subgoal frames reads the cube within B_cube = 2.397 cm in median (reported beside it:
    per subgoal, and on the re-rendered reset frames)."""
    from embodied_jepa import so101_wm as sw

    goals = load_goals(args, hy.LEDGER_COHORTS)
    report["goals_sha256"], report["cohort"] = goals["sha256"], goals["report"]["cohort"]
    shared = rt.shared_root()
    rr = json.loads((shared / cl.TASK100_READOUTS / "report.json").read_text())
    path = shared / cl.TASK100_READOUTS / cl.RENC_FILE
    if cl.sha256_file(path) != rr["files_sha256"][cl.RENC_FILE]:
        raise hy.GuardError("G-readouts: renc_pretrained.npz differs from TASK-100's report")
    fits = sw.load_readouts(path)
    moments = rt.load_moments()
    a = goals["arrays"]
    x = rt.standardise(moments, a["latents_raw"].reshape(-1, cl.TOKENS, cl.WIDTH))
    err = sw.label_errors("cube", sw.apply_readout(fits["cube"], x), a["cube_cm"].reshape(-1, 3))
    bar = sw.frozen_choices()["bars"]["B_cube"]
    feat = rt.LiveFeaturiser(args.device)
    reset_err = []
    for seed in [int(s) for s in a["seeds"]]:
        robot = rt.make_robot(True)
        try:
            rt.lift_task(robot).reset(seed)
            frames = {c: robot.sim.render(c)[None] for c in cl.CAMERAS}
            truth = robot.sim.data.body("cube").xpos.copy() * 100.0
        finally:
            robot.close()
        _, z = feat.encode(frames)
        reset_err.append(
            float(sw.label_errors("cube", sw.apply_readout(fits["cube"], z), truth[None])[0])
        )
    median = float(np.median(err))
    return {
        "frames": int(len(err)),
        "median_cm": median,
        "B_cube": bar,
        "per_subgoal_median_cm": {
            str(k + 1): float(np.median(err.reshape(-1, 4)[:, k])) for k in range(4)
        },
        "reset_frame_median_cm": float(np.median(reset_err)),
        "reset_frames": len(reset_err),
        "passes": bool(median <= bar),
    }


def check_projector(args, report) -> dict:
    """7.3.3 = TASK-101's 7.2.3 on P's reset states: 1 000 random 16-step sequences; the
    vectorised chained projection equals ``preview`` chained step by step and the applied actions
    ``execute`` returns."""
    goals = load_goals(args, ("P", "Pp"))
    report["goals_sha256"], report["cohort"] = goals["sha256"], goals["report"]["cohort"]
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    rng = np.random.default_rng([hy.SALTS["rand"], 7])  # plumbing draws for the check only
    preview_equal = execute_equal = stopped = compared = 0
    robot = rt.make_robot(False)
    task = rt.lift_task(robot)
    params = cl.PreviewParams.of(robot)
    try:
        for i in range(args.sequences):
            task.reset(seeds[i % len(seeds)])
            req = rng.uniform(-1, 1, (16, 6)).astype(np.float32)
            if i % 4 == 0:
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


def check_gloc(args, report) -> dict:
    """7.3.4: on 256 stored val frames (evenly spaced over the val windows), G_loc's live path
    (``act``: the non-latent inputs rebuilt from the measured joints, velocities, targets and the
    clock, one command at a time) equals its batched offline prediction within 1e-4, for every
    G_loc seed."""
    from embodied_jepa import so101_grasp_policy as gp

    sel = hr.load_gloc_selection(args.gloc)
    windows = Path(args.windows)
    moments = rt.load_moments()
    val = hr.load_split_frames(windows, "val", moments, in_memory=False)
    idx = np.linspace(0, len(val) - 1, 256).astype(np.int64)
    z = moments.apply(np.asarray(val.raw16[idx], np.float32))
    x35 = val.x35[idx]
    out = {"frames": int(len(idx)), "seeds": {}}
    for name, entry in sel["checkpoints"].items():
        policy = gp.load_checkpoint(entry["path"], args.device, entry["sha256"])
        batched = policy.predict(z, x35)
        live = np.stack(
            [
                policy.act(
                    z[i], x35[i, :6], x35[i, 6:12], x35[i, 12:18], int(round(x35[i, 18] * 100))
                )[1]
                for i in range(len(idx))
            ]
        )
        clock_ok = bool(
            np.array_equal(
                np.stack([hy.nonlatent(r[:6], r[6:12], r[12:18], round(r[18] * 100)) for r in x35]),
                x35,
            )
        )
        diff = float(np.abs(live - batched).max())
        out["seeds"][name] = {"max_abs": diff, "inputs_rebuilt_equal": clock_ok}
    out["passes"] = all(
        v["max_abs"] <= 1e-4 and v["inputs_rebuilt_equal"] for v in out["seeds"].values()
    )
    return out


DETERMINISM_CONFIG = "H8-mean"  # plumbing values for 7.3.5 (τ_h and the configuration are set
DETERMINISM_TAU = hy.TAU_FALLBACK  # by P and A later; TASK-101 7.2.4 ran at H8-mean too)


def check_determinism(args, report) -> dict:
    """7.3.5, conditional form (TASK-101 7.2.4 as A1 amended it): one H and one S+G episode, each
    run twice (pool of 1), and four H episodes in a pool of 1 and in the pool size used, on debug
    seeds. 7.3.6 (timing) is read from the same episodes and written as its own ledger entry."""
    goals = load_goals(args, ("debug",))
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    base = Path(args.output)
    runs = {}

    def run(tag, arm, slots, workers):
        folder = base / tag
        folder.mkdir(parents=True)
        specs = [
            {
                "arm": arm,
                "seed": seeds[s],
                "slot": s,
                "config": DETERMINISM_CONFIG if arm == "H" else None,
                "tau_h": DETERMINISM_TAU,
                "level": 0.0 if arm == "S+G" else None,
                "keep_frames": False,
                "output": str(folder),
            }
            for s in slots
        ]
        t0 = time.monotonic()
        res = hr.run_pool(
            specs,
            goals_dir=args.goals,
            goals_sha256=goals["sha256"],
            device=args.device,
            workers=workers,
            gloc_file=args.gloc,
        )
        runs[tag] = {"results": res, "wall_seconds": time.monotonic() - t0, "workers": workers}
        return {r["seed"]: r for r in res}

    h1 = run("H-a", "H", [0], 1)
    h2 = run("H-b", "H", [0], 1)
    s1 = run("SG-a", "S+G", [0], 1)
    s2 = run("SG-b", "S+G", [0], 1)
    p1 = run("pool-1", "H", [0, 1, 2, 3], 1)
    pk = run(f"pool-{args.workers}", "H", [0, 1, 2, 3], args.workers)
    pk_tag = f"pool-{args.workers}"

    def pair(x, y, tx, ty, seed):
        return hr.pair_divergence(
            base / tx / f"{seed}.npz", base / ty / f"{seed}.npz", x[seed], y[seed]
        )

    pairs = {
        "H_rerun": pair(h1, h2, "H-a", "H-b", seeds[0]),
        "SG_rerun": pair(s1, s2, "SG-a", "SG-b", seeds[0]),
        **{f"pool_{s}": pair(p1, pk, "pool-1", pk_tag, s) for s in seeds[:4]},
    }
    result = {
        "config": DETERMINISM_CONFIG,
        "tau_h": DETERMINISM_TAU,
        "pairs": pairs,
        "literal_identical": all(v["bit_identical"] for v in pairs.values()),
        "conditional_determinism": all(v["conditional_determinism"] for v in pairs.values()),
        "workers": args.workers,
        "runs": {
            k: {
                "wall_seconds": v["wall_seconds"],
                "workers": v["workers"],
                "episodes": [
                    {
                        x: r.get(x)
                        for x in (
                            "arm",
                            "seed",
                            "termination",
                            "commands",
                            "handover",
                            "phase1",
                            "phase2",
                            "episode_seconds",
                            "step_seconds_max",
                        )
                    }
                    for r in v["results"]
                ],
            }
            for k, v in runs.items()
        },
    }
    result["passes"] = result["conditional_determinism"]
    report["timing"] = timing_from_runs(runs, args.workers)
    return result


def timing_from_runs(runs: dict, workers: int) -> dict:
    """7.3.6: seconds per command in phase 1 and phase 2 (median and 95th percentile over the
    episodes' per-phase medians and 95th percentiles, all runs), the worker count, the pool's
    lock seconds per command, the projection of P, A, D and S, and whether every command
    finished inside the embodiment's 30 s deadline."""
    worst = 0.0
    per_phase = {1: [], 2: []}
    for v in runs.values():
        for r in v["results"]:
            for phase in (1, 2):
                ph = r[f"phase{phase}"]
                if ph["commands"]:
                    per_phase[phase].append(ph)
            worst = max(worst, r["step_seconds_max"] or 0.0)
    pool = runs[f"pool-{workers}"]
    n_parallel = min(workers, len(pool["results"]))

    def med(phase, key, results=None):
        rows = (
            per_phase[phase]
            if results is None
            else [r[f"phase{phase}"] for r in results if r[f"phase{phase}"]["commands"]]
        )
        return float(np.median([x[key] for x in rows])) if rows else None

    m1, m2 = med(1, "median", pool["results"]), med(2, "median", pool["results"])
    predicted = sum(
        r["phase1"]["commands"] * (m1 or 0.0) + r["phase2"]["commands"] * (m2 or 0.0)
        for r in pool["results"]
    ) / max(n_parallel, 1)
    overhead = pool["wall_seconds"] / predicted if predicted > 0 else None
    s1 = (m1 or 0.0) / max(n_parallel, 1) * (overhead or 1.0)
    s2 = (m2 or 0.0) / max(n_parallel, 1) * (overhead or 1.0)
    full = hy.T1 * s1 + (hy.MAX_COMMANDS - hy.T1) * s2  # a CEM approach then G_loc
    gloc = hy.MAX_COMMANDS * s2  # G_loc (or a cheaper scripted phase) for every command
    proj = {
        "lock_seconds_per_command": {"phase1": s1, "phase2": s2, "pool_overhead_factor": overhead},
        "P_gpu_h": 128 * gloc / 3600,
        "A_gpu_h": 128 * hy.T1 * s1 / 3600,
        "D_gpu_h": 64 * (4 * full + 2 * gloc) / 3600,
        "S_gpu_h": 128 * (6 * full + 2 * gloc) / 3600,
        "note": "upper bounds: every episode runs its full 200-command budget and every CEM "
        "approach its 60 commands; measured on the determinism check's pool runs",
    }
    return {
        "phase1_seconds_per_command": {"median": med(1, "median"), "p95": med(1, "p95")},
        "phase2_seconds_per_command": {"median": med(2, "median"), "p95": med(2, "p95")},
        "workers": workers,
        "pool_lock_seconds_per_command": pool["wall_seconds"]
        / max(sum(r["commands"] for r in pool["results"]), 1),
        "projection": proj,
        "max_step_seconds": worst,
        "passes": bool(worst < 30.0),
    }


CHECKS = {
    "latent": check_latent,
    "renc": check_renc,
    "projector": check_projector,
    "gloc": check_gloc,
    "determinism": check_determinism,
}


def _ledger(args, report, check, passes, sha) -> None:
    hy.ledger_append(
        ledger_path(args),
        hy.ledger_entry(
            check=check,
            passes=passes,
            revision=report["revision"],
            dirty=report["tracked_tree_dirty"],
            finished_utc=report["finished_utc"],
            goals_sha256=report.get("goals_sha256"),
            cohort_name=report.get("cohort"),
            report_sha256=sha,
        )
        | {"report": str(Path(args.output) / "report.json")},
    )


def cmd_check(args) -> int:
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    report = hr.header("check", check=args.check, device=args.device)
    guards(report, False)
    if args.device == "cuda":
        gpu_prelude(report, args.device)
    t0 = time.monotonic()
    report["result"] = CHECKS[args.check](args, report)
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = hr.utc()
    sha = hr.write_json(out / "report.json", report)
    _ledger(args, report, args.check, report["result"].get("passes"), sha)
    if args.check == "determinism":  # 7.3.6 from the same episodes, its own ledger entry
        _ledger(args, report, "timing", report["timing"]["passes"], sha)
    hr.log(f"check {args.check}: passes={report['result'].get('passes')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=sorted(hy.COHORTS))
    g.add_argument("--output", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--gated", action="store_true")
    e = sub.add_parser("episodes")
    e.add_argument("--cohort", required=True, choices=sorted(hy.COHORTS))
    e.add_argument("--goals", required=True)
    e.add_argument("--goals-sha256", required=True)
    e.add_argument("--arms", required=True)
    e.add_argument("--configs", default=None)
    e.add_argument("--levels", default=None)
    e.add_argument("--tau-file", default=None)
    e.add_argument("--config-file", default=None)
    e.add_argument("--tau-h", type=float, default=None)
    e.add_argument("--config", default=None)
    e.add_argument("--gloc", default=None)
    e.add_argument("--slots", default=None)
    e.add_argument("--evidence", type=int, default=hy.EVIDENCE_FRAMES_EPISODES)
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--device", default="cuda")
    e.add_argument("--output", required=True)
    e.add_argument("--gated", action="store_true")
    e.add_argument("--ledger", default=None)
    c = sub.add_parser("check")
    c.add_argument("check", choices=sorted(CHECKS))
    c.add_argument("--goals", default=None)
    c.add_argument("--goals-sha256", default=None)
    c.add_argument("--gloc", default=None)
    c.add_argument("--windows", default=None)
    c.add_argument("--device", default="cuda")
    c.add_argument("--sequences", type=int, default=1000)
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--ledger", default=None)
    c.add_argument("--output", required=True)
    args = p.parse_args(argv)
    return {"goals": cmd_goals, "episodes": cmd_episodes, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
