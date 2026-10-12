"""TASK-090 D-A2: a development-only diagnosis of A2 (DECISIONS R27.36-R27.41). Not gated; not
frozen code; its counts are development readings, not results.

It runs the frozen code of the TASK-090 run tree (``--run-tree``, which must be a clean tree at
``53f0b72``) and changes no frozen module:

* ``roots``: 64 development roots on seeds 905000-905063 with the frozen sampler
  (``zg.build_root``: the A-prime root builder, its rejection rule and branches unchanged).
  Renders with EGL, so it runs under ``scripts/gpu_run.sh``.
* ``measure``: for the nine G checkpoints (TASK-090's six and TASK-087's three), at R1 and R2 of
  every root, TASK-089's A2 planner model (``zgr.wrap`` with setting ``step-mean``,
  lambda = 2.0552) and a CEM planner built exactly as ``zgr.make_planner`` builds it (``zs.CEM``,
  TASK-088's bounds) but with an explicit seed, keeping the whole 8-step plan:
  (a) close@R1 (and lift@R2, reported) at the A2 seed ``zs.cem_seed(root)``;
  (b) the first plan step k at R1 whose right-grasp command is > 0.5, at the A2 seed;
  (c) close@R1 under four further seeds ``9007 * 10**7 + j * 10**6 + root`` (j = 1..4);
  (d) the mean right-grasp channel per plan step at R1, at the A2 seed.
  At the A2 seed the plan's first command must equal ``zgr.first_planned``'s, bitwise, on every
  root (R1 and R2) of every model; otherwise it stops (R27.37).
* ``reading``: R27.40's reading from the measure report, in its declared order.

    D=/home/huhn/develop/emai/worktrees/task090-run
    MUJOCO_GL=egl DA2_RUN_TREE=$D scripts/gpu_run.sh --wait --min-free-gib 8 --board -- \\
        $D/.venv/bin/python scripts/dev_task090_da2.py roots --out <evidence>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

RUN_TREE = Path(os.environ.get("DA2_RUN_TREE", "/home/huhn/develop/emai/worktrees/task090-run"))
sys.path.insert(0, str(RUN_TREE / "src"))

REVISION = "53f0b72f75b61031c9bbba714b3e84c32b194925"
SEEDS = range(905000, 905064)
SALT = 9007
EXTRA = (1, 2, 3, 4)
BAR = 48
FAILING = ("G-v2g-90100", "G-v1p-90100", "G-v1p-90102")
MODELS = [("play-v2g", s) for s in (90100, 90101, 90102)]
MODELS += [("play-v1p", s) for s in (90100, 90101, 90102)]
MODELS += [("play-v1", s) for s in (87100, 87101, 87102)]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_module():
    """The run tree's ``scripts/run_task090.py`` (its guards, report and helpers)."""
    spec = importlib.util.spec_from_file_location(
        "run_task090_frozen", RUN_TREE / "scripts" / "run_task090.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extra_seed(j: int, root: int) -> int:
    return SALT * 10**7 + j * 10**6 + int(root)


def _worker(seeds):
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_grasp as zg

    robot = pc.make_play_robot()
    out = [zg.build_root(robot, s) for s in seeds]
    robot.close()
    return out


def report_base(r, args):
    rep = r.base_report(None, clean=True)
    if rep["revision"] != REVISION:
        raise SystemExit(f"run tree is at {rep['revision']}, not {REVISION}")
    rep |= {
        "development": "TASK-090 D-A2 (R27.36-R27.41); not gated; not results",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha(__file__),
        "run_tree": str(RUN_TREE),
        "argv": sys.argv,
    }
    return rep


def cmd_roots(args) -> int:
    r = run_module()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / "roots-da2.json", out / "roots-da2.npz"
    for p in (jpath, npath):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    rep = report_base(r, args)
    seeds = list(SEEDS)
    t0 = time.time()
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_worker, [c for c in chunks if c])
    roots = sorted((x for p in parts for x in p), key=lambda x: x["seed"])
    arrays, meta = {}, []
    for x in roots:
        s = x["seed"]
        row = {k: x[k] for k in ("seed", "attempt", "layout", "target", "yaw", "z0", "rejected")}
        for k, g in enumerate(x["subgoals"]):
            arrays[f"{s}/subgoal{k}/frame"] = g["frame"]
            arrays[f"{s}/subgoal{k}/state28"] = g["state28"]
        for root, rec in x["roots"].items():
            arrays[f"{s}/{root}/frame"] = rec["frame"]
            arrays[f"{s}/{root}/state28"] = rec["state28"]
        meta.append(row)
    np.savez_compressed(npath, **arrays)
    rep |= {
        "seconds": time.time() - t0,
        "seeds": [seeds[0], seeds[-1]],
        "rejections": int(sum(len(m["rejected"]) for m in meta)),
        "roots": meta,
        "arrays_sha256": sha(npath),
    }
    r.write_json(jpath, rep)
    print(json.dumps({"roots": len(meta), "rejections": rep["rejections"]}))
    return 0


def plan_actions(pm, seed, rl, rs, gl, gs):
    """The whole plan of one CEM call, built as ``zgr.make_planner`` builds it with ``seed``."""
    from embodied_jepa import grasp_corpus_wm as gc
    from embodied_jepa import zero_shot as zs
    from embodied_jepa import zero_shot_grasp_runtime as zgr
    from embodied_jepa.planning import CEMConfig, CEMPlanner

    model, horizon = zgr.wrap(pm, gc.A_PRIME_SETTING)
    cfg = zs.CEM
    planner = CEMPlanner(
        CEMConfig(
            horizon=horizon,
            samples=cfg["samples"],
            iterations=cfg["iterations"],
            elites=cfg["elites"],
            minimum_std=cfg["minimum_std"],
            seed=int(seed),
            lower_bounds=tuple(zs.FIXED_LOWER.tolist()),
            upper_bounds=tuple(zs.FIXED_UPPER.tolist()),
        )
    )
    plan = planner.plan(model, model.encode(rl, rs), model.encode_goal(gl, gs))
    return plan.actions[0].astype(np.float32)


def cmd_measure(args) -> int:
    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm as gc
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import zero_shot as zs
    from embodied_jepa import zero_shot_grasp as zg
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    r = run_module()
    folder = Path(args.out)
    out = folder / "measure-da2.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rep = report_base(r, args)
    if gc.A_PRIME_SETTING != "step-mean" or abs(zs.LAMBDA - 2.0552) > 1e-12:
        raise SystemExit("A2 setting or lambda is not R27.37's")
    roots_rep = json.loads((folder / "roots-da2.json").read_text())
    npath = folder / "roots-da2.npz"
    if sha(npath) != roots_rep["arrays_sha256"] or roots_rep["revision"] != REVISION:
        raise SystemExit("roots arrays or revision mismatch")
    if [x["seed"] for x in roots_rep["roots"]] != list(SEEDS):
        raise SystemExit("roots are not exactly 905000-905063")
    data = np.load(npath)
    devices.configure_determinism("cuda", strict=False)
    enc_fn = gr.encoder("cuda")
    enc = {}
    for x in roots_rep["roots"]:
        s = x["seed"]
        for root in zg.ROOTS:
            enc[f"{s}/{root}"] = enc_fn(data[f"{s}/{root}/frame"])[0]
            enc[f"{s}/goal/{root}"] = enc_fn(data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/frame"])[0]
    pins = r._feature_pins(None)
    rep |= {
        "roots_json_sha256": sha(folder / "roots-da2.json"),
        "roots_arrays_sha256": roots_rep["arrays_sha256"],
        "feature_reports_sha256": pins,
        "lambda": zs.LAMBDA,
        "setting": gc.A_PRIME_SETTING,
        "cem": zs.CEM,
        "extra_seed_rule": f"{SALT} * 10**7 + j * 10**6 + root, j in {list(EXTRA)}",
    }
    gd = zg.GRASP_DIM
    t0 = time.time()
    results = {}
    for corpus, seed in MODELS:
        if corpus == "play-v1":
            model, info = gr.load_v1_model(seed, "cuda", arm="G")
        else:
            model, info = gr.load_new_model(
                gr.model_dir(r.ROOT, "G", corpus, seed), "cuda", feature_reports=pins
            )
        name = f"G-{gc.CORPUS_TAG.get(corpus, 'v1')}-{seed}"
        pm = gr.planner_model(model, w_pose=1.0)
        plans_r1, first_r2, extra = [], [], {j: [] for j in EXTRA}
        mismatches = []
        tm = time.time()
        for x in roots_rep["roots"]:
            s = x["seed"]
            args_root = {}
            for root in zg.ROOTS:
                args_root[root] = (
                    enc[f"{s}/{root}"],
                    data[f"{s}/{root}/state28"],
                    enc[f"{s}/goal/{root}"],
                    data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"],
                )
            for root in zg.ROOTS:
                a2 = zs.cem_seed(s)
                full = plan_actions(pm, a2, *args_root[root])
                ref = zgr.first_planned(pm, gc.A_PRIME_SETTING, s, *args_root[root])
                if not np.array_equal(full[0], ref):
                    mismatches.append([s, root])
                if root == "R1":
                    plans_r1.append(full.tolist())
                else:
                    first_r2.append(full[0].tolist())
            if mismatches:
                break
            for j in EXTRA:
                extra[j].append(plan_actions(pm, extra_seed(j, s), *args_root["R1"])[0].tolist())
        if mismatches:
            rep |= {
                "stopped": "first-command equality check failed",
                "model": name,
                "mismatches": mismatches,
            }
            r.write_json(out, rep)
            print(json.dumps({"stopped": name, "mismatches": mismatches}))
            return 3
        p1 = np.array(plans_r1)  # [64, 8, 14]
        r2 = np.array(first_r2)
        closes = p1[:, :, gd] > 0.5
        k = [int(np.argmax(c)) + 1 if c.any() else None for c in closes]
        m = zg.a2_measures(p1[:, 0], r2)
        close_extra = {str(j): int((np.array(extra[j])[:, gd] > 0.5).sum()) for j in EXTRA}
        close_all = [int(m["close_R1"].sum())] + list(close_extra.values())
        results[name] = {
            "model": info,
            "a_close_R1": int(m["close_R1"].sum()),
            "a_lift_R2_reported": int(m["lift_R2"].sum()),
            "b_first_close_step": k,
            "b_close_by": {
                str(kk): int(sum(1 for v in k if v is not None and v <= kk)) for kk in (2, 4, 8)
            },
            "b_never_close": int(sum(v is None for v in k)),
            "c_close_R1_extra_seeds": close_extra,
            "c_close_R1_all_five": close_all,
            "c_spread": int(max(close_all) - min(close_all)),
            "d_mean_grasp_per_step": [float(v) for v in p1[:, :, gd].mean(0)],
            "plans_R1_a2_seed": p1.tolist(),
            "first_R2_a2_seed": r2.tolist(),
            "first_R1_extra_seeds": {str(j): extra[j] for j in EXTRA},
            "seconds": time.time() - tm,
        }
        print(
            f"{name}: a {results[name]['a_close_R1']} b {results[name]['b_close_by']} "
            f"c {close_all} d {[round(v, 2) for v in results[name]['d_mean_grasp_per_step']]}",
            flush=True,
        )
        del model
    rep |= {
        "first_command_check": "bitwise equal to zgr.first_planned on every root and model",
        "results": results,
        "reading": reading(results),
        "seconds": time.time() - t0,
        "finished_utc": r.utc(),
    }
    r.write_json(out, rep)
    print(json.dumps(rep["reading"], indent=1))
    return 0


def reading(results) -> dict:
    """R27.40, in its declared order (D-REPLICATE-FAIL first, set at review)."""
    a = {n: v["a_close_R1"] for n, v in results.items()}
    others = [n for n in a if n not in FAILING]
    replicate_fail = any(a[n] >= BAR for n in FAILING) or any(a[n] < BAR for n in others)
    cem = any(results[n]["c_spread"] >= 16 for n in FAILING)
    timing = all(results[n]["b_close_by"]["4"] >= BAR for n in FAILING)
    cost = any(results[n]["b_close_by"]["8"] < BAR for n in FAILING)
    if replicate_fail:
        row = "D-REPLICATE-FAIL"
    elif cem:
        row = "D-CEM"
    elif timing:
        row = "D-TIMING"
    elif cost:
        row = "D-COST"
    else:
        row = "D-NONE"
    return {
        "row": row,
        "checks": {
            "replicate_fail": replicate_fail,
            "cem_spread_ge_16_any_failing": cem,
            "timing_all_failing_close_by_4_ge_48": timing,
            "cost_any_failing_close_by_8_lt_48": cost,
        },
        "failing_models": list(FAILING),
        "bar": BAR,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ro = sub.add_parser("roots")
    ro.add_argument("--out", required=True)
    ro.add_argument("--workers", type=int, default=8)
    me = sub.add_parser("measure")
    me.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"roots": cmd_roots, "measure": cmd_measure}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
