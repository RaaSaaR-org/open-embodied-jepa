"""TASK-090: Stage 0's checks, featurisation, Stage E, A-K0, Stage A′ and the closed loop
(protocol docs/experiments/grasp_play_corpus_v1.md). Subcommands; every GPU use is one job under
``scripts/gpu_run.sh --wait --min-free-gib 8 --board``:

* ``verify --output FILE``: §12's path and sha256 checks (the links resolve to the data disk;
  ``play-v1``, TASK-087's feature report, store files, projection and 18 checkpoints) and §2.2's
  bitwise moments check against G-87100 (CPU).
* ``power --output FILE``: §7.2's power estimate and rule (CPU).
* ``featurise --corpus C --output DIR``: a new collection's train and val features with TASK-087's
  pinned projection (``--debug-corpus DIR`` for a debug collection; test is never featurised).
* ``stage-e --output FILE``: §6.1 on the common val windows.
* ``roots --cohort C --output DIR`` and ``ceiling --cohort k0-roots``: A-K0 (TASK-089's Part A
  roots and simulator ceiling; no model runs).
* ``a-prime --cohort a-roots --roots DIR --k0 FILE --output DIR``: Stage A′ (§6.2).
* ``goals --cohort C --output DIR``; ``run --cohort C --arm A [--model-seed S]``;
  ``video --result FILE --goals DIR --seed N --output FILE.mp4``: the closed loop (§7).

Non-debug cohorts need a clean tracked tree; every report records the revision; every file is
refused if it exists.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grounded_wm as gw  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402

DISK_FLOOR_GIB = 10.0
DATA_DISK = Path("/mnt/data/emai/projects/open-embodied-jepa")
CORPUS_DIRS = {"play-v1": "data/play-v1", "play-v2g": "data/play-v2g", "play-v1p": "data/play-v1p"}
# §7.1: the closed-loop arms; model arms -> (model source, the seeds a gated cohort admits)
MODEL_ARMS = gc.CLOSED_LOOP_ARMS
BD_SEEDS = gc.BD_SEEDS
BS_SEEDS = gc.BS_SEEDS
DETERMINISM = ("M-v2g", 90100, 4)


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def base_report(cohort=None, *, clean=False, gpu=True) -> dict:
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    report = {
        "task": gc.TASK,
        "protocol": gc.PROTOCOL,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "cohort": cohort,
    }
    rt.assert_local_import(ROOT, report)
    if (clean or (cohort is not None and not cohort.startswith("debug"))) and dirty:
        raise rt.GuardError("G-clean: non-debug stages run from a clean tracked tree")
    free = min(shutil.disk_usage("/").free, shutil.disk_usage(ROOT / "data").free) / 2**30
    report["disk_free_gib"] = free
    if free < DISK_FLOOR_GIB:
        raise rt.GuardError(f"G-disk: {free:.1f} GiB free < {DISK_FLOOR_GIB}")
    if gpu:
        if os.environ.get(rt.GPU_LOCK_ENV) != "1":
            raise rt.GuardError("G-gpu-lock: run through scripts/gpu_run.sh")
        report["gpu_lock_held"] = True
    report["mujoco_gl"] = os.environ.get("MUJOCO_GL")
    return report


def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise rt.GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


def sha(path) -> str:
    return gd.sha256_file(path)


# ----- §12 and §2.2: verify ----------------------------------------------------------------------
def cmd_verify(args) -> int:
    import torch  # noqa: F401  (checkpoint buffers)

    from embodied_jepa import grasp_corpus_wm_runtime as gr

    report = base_report(gpu=False)
    out = Path(args.output)
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    checks = {}
    # the links resolve to the data disk, and it is rotational
    links = {}
    for name in ("data", "checkpoints", "outputs"):
        for base in (ROOT, Path(args.main_checkout)):
            p = base / name
            if not p.is_symlink():
                continue
            target = p.resolve()
            links[str(p)] = {"target": str(target), "on_data_disk": DATA_DISK in target.parents}
    report["links"] = links
    dev = subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", "--target", str(DATA_DISK)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    block = Path(dev).name.rstrip("0123456789")
    rota = Path(f"/sys/block/{block}/queue/rotational")
    report["data_disk"] = {
        "device": dev,
        "rotational": rota.read_text().strip() == "1" if rota.exists() else None,
    }
    checks["links_on_data_disk"] = bool(links) and all(v["on_data_disk"] for v in links.values())
    # play-v1: corpus.json pinned, every shard's files re-hashed by DatasetStore
    t0 = time.monotonic()
    report["play_v1"] = gd.verify_corpus(ROOT / CORPUS_DIRS["play-v1"])
    report["play_v1_seconds"] = time.monotonic() - t0
    checks["play_v1"] = True
    # TASK-087's feature report, store files and projection
    feat = gr.V1_FEATURES
    checks["feature_report"] = sha(feat / "report.json") == gr.V1_FEATURES_REPORT_SHA256
    frep = json.loads((feat / "report.json").read_text())
    report["store_files"] = gd.verify_store_files(feat, frep["files_sha256"], ("train", "val"))
    checks["store_files"] = True
    proj_file = sha(feat / "projection.npz")
    checks["projection_file"] = proj_file == frep["projection"]["file_sha256"]
    gd.load_projection(feat / "projection.npz", gr.PROJECTION_SHA256)
    checks["projection_arrays"] = True
    # the 18 checkpoints: model.pt against its report and the evidence copy of the report
    evidence = Path(args.task087_evidence)
    ckpts = {}
    for seed in (87100, 87101, 87102):
        for arm in gw.ARMS:
            name = f"{arm}-{seed}"
            folder = gr.V1_CHECKPOINTS / name
            rep = json.loads((folder / "report.json").read_text())
            ev = evidence / f"train_report_{name}.json"
            row = {
                "model_pt": sha(folder / "model.pt"),
                "recorded": rep["checkpoint_sha256"],
                "report_equals_evidence": sha(folder / "report.json") == sha(ev),
            }
            row["ok"] = row["model_pt"] == row["recorded"] and row["report_equals_evidence"]
            ckpts[name] = row
    report["checkpoints"] = ckpts
    checks["checkpoints"] = all(r["ok"] for r in ckpts.values()) and len(ckpts) == 18
    # §2.2: the pinned moments equal G-87100's buffers bitwise
    report["moments"] = gr.check_pinned_moments()
    checks["moments_bitwise"] = report["moments"]["bitwise_equal"]
    from embodied_jepa import grounded_wm_model as gm

    checks["implementation_hash"] = gm.implementation_sha256() == gr.IMPLEMENTATION_SHA256
    report["checks"] = checks
    report["all_pass"] = all(checks.values())
    report["finished_utc"] = utc()
    write_json(out, report)
    log(json.dumps(checks))
    return 0 if report["all_pass"] else 1


# ----- §7.2: power -------------------------------------------------------------------------------
def cmd_power(args) -> int:
    report = base_report(gpu=False)
    table, rows = {}, []
    for n in (64, 128):
        table[n] = {}
        for sd in gc.POWER_SDS:
            r = gc.power(n, sd, sims=args.sims)
            table[n][sd] = r["power"]
            rows.append(r)
            log(json.dumps(r))
    report |= {
        "alternative": gc.POWER_ALTERNATIVE,
        "rows": rows,
        "rule": gc.power_rule(table),
        "finished_utc": utc(),
    }
    write_json(args.output, report)
    log(json.dumps(report["rule"]))
    return 0


# ----- §2.2: featurisation -----------------------------------------------------------------------
def cmd_featurise(args) -> int:
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import play_corpus_grasp as pg

    debug = args.debug_corpus is not None
    report = base_report(None if debug else "features", clean=not debug)
    rt.gpu_guard(report, min_free_gib=4.0, require_lock=True)
    corpus = Path(args.debug_corpus) if debug else ROOT / CORPUS_DIRS[args.corpus]
    out = Path(args.output) if debug else gr.new_features(ROOT, args.corpus)
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    record = json.loads((corpus / "corpus.json").read_text())
    if record.get("task") != gc.TASK or bool(record.get("debug")) != debug:
        raise rt.GuardError("G-corpus: not a TASK-090 collection of this kind")
    if not debug and record["corpus"] != args.corpus:
        raise rt.GuardError("G-corpus: another corpus")
    report["corpus"] = record["corpus"]
    report["corpus_check"] = gd.verify_corpus(corpus, pinned=None)
    eps = gd.corpus_episodes(corpus, check_counts=False)
    policy = "grasp" if record["corpus"] in ("play-v2g", "debug-grasp") else "v0"
    for split in ("train", "val"):
        for _, _, seed in eps[split]:
            pg.check_collection_seed(policy, seed, debug=debug)
    featuriser, projection = gr.featuriser_and_projection()
    report["encoder_digest"] = featuriser.digest
    report["projection_sha256"] = gr.PROJECTION_SHA256
    out.mkdir(parents=True)
    t0 = time.monotonic()
    files = {}
    for split in ("train", "val"):  # test is never featurised (§2.2)
        files[split] = gr.featurise_split(
            corpus, eps[split], split, out, featuriser, projection, args.workers, log
        )
    report |= {
        "files_sha256": files,
        "episodes": {k: len(eps[k]) for k in ("train", "val")},
        "seconds": time.monotonic() - t0,
        "debug": debug,
        "finished_utc": utc(),
        "outcome": "F-DONE",
    }
    write_json(out / "report.json", report)
    log(f"F-DONE {report['episodes']} in {report['seconds']:.0f} s")
    return 0


# ----- §6.1: Stage E -----------------------------------------------------------------------------
def _feature_pins(debug_dirs=None) -> dict:
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    pins = {"play-v1": gr.V1_FEATURES_REPORT_SHA256}
    for corpus in gc.CORPORA:
        folder = Path(debug_dirs[corpus]) if debug_dirs else gr.new_features(ROOT, corpus)
        pins[corpus] = sha(folder / "report.json")
    return pins


def _val_store(debug_dirs=None):
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    dirs = debug_dirs or {c: gr.new_features(ROOT, c) for c in gc.CORPORA}
    return gc.CombinedStore(
        [(gr.V1_FEATURES, "val"), (dirs["play-v2g"], "val"), (dirs["play-v1p"], "val")], mmap=False
    )


def _val_window_labels(val, corpus_dirs) -> dict:
    """Per frame of the common val store: any lift state, any grasp contact (from sidecars)."""
    lift = np.zeros(len(val.state), bool)
    contact = np.zeros(len(val.state), bool)
    for row in val.table:
        corpus = Path(corpus_dirs[row["store"]])
        manifest = json.loads((corpus / row["shard"] / "meta" / "jepa_manifest.json").read_text())
        meta = next(e for e in manifest["episodes"] if e["episode_id"] == row["episode_id"])
        meta = meta["metadata"]
        path = corpus / row["shard"] / meta["sidecar"]
        if sha(path) != meta["sidecar_sha256"]:
            raise rt.GuardError(f"G-sidecar: {path}")
        with np.load(path) as d:
            lab = gc.frame_labels({k: d[k] for k in d.files}, gc.pc.on_table_mask(meta["layout"]))
        lo, n = row["offset"], row["frames"]
        lift[lo : lo + n] = lab["lift"].any(1)
        contact[lo : lo + n] = lab["contact"].any(1)
    return {"lift": lift, "contact": contact}


def _models_for_e(debug):
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    out = []
    seeds = (gc.DEBUG_MODEL_SEEDS[0],) if debug else gc.MODEL_SEEDS
    for corpus in gc.CORPORA:
        for seed in seeds:
            out.append(("G", corpus, seed))
        out.append(("N", corpus, seeds[0] if debug else gc.N_SEED))
    for seed in (87100, 87101, 87102):
        out.append(("G", "play-v1", seed))
    return out, gr


def cmd_stage_e(args) -> int:
    from embodied_jepa import devices
    from embodied_jepa import grounded_wm_model as gm

    debug = args.debug_features is not None
    report = base_report("debug-e" if debug else "stage-e")
    rt.gpu_guard(report, min_free_gib=4.0, require_lock=True)
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    debug_dirs = dict(zip(gc.CORPORA, args.debug_features, strict=True)) if debug else None
    pins = _feature_pins(debug_dirs)
    report["feature_reports_sha256"] = pins
    val = _val_store(debug_dirs)
    roots, episode = val.roots()
    horizons = gc.E_HORIZONS
    start, commands, states, targets = val.root_arrays(roots, horizons=horizons)
    commands = commands[:, : max(horizons)]
    wrong = gw.wrong_table(episode)
    weights = gw.cluster_bootstrap(episode)
    report["val"] = {"episodes": len(val.table), "roots": int(len(roots)), "parts": val.parts}
    corpus_dirs = {}
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    corpus_dirs[str(Path(gr.V1_FEATURES))] = ROOT / CORPUS_DIRS["play-v1"]
    for corpus in gc.CORPORA:
        folder = Path(debug_dirs[corpus]) if debug else gr.new_features(ROOT, corpus)
        src = json.loads((folder / "report.json").read_text())["argv"]
        corpus_dirs[str(folder)] = (
            Path(src[src.index("--debug-corpus") + 1]) if debug else ROOT / CORPUS_DIRS[corpus]
        )
    labels = _val_window_labels(val, corpus_dirs)
    win = roots[:, None] + np.arange(gw.TRAIN_HORIZON + 1)
    classes = {"lift": labels["lift"][win].any(1), "contact": labels["contact"][win].any(1)}
    report["val"]["windows_lift"] = int(classes["lift"].sum())
    report["val"]["windows_contact"] = int(classes["contact"].sum())
    models, _ = _models_for_e(debug)
    errors, results = {}, {}
    for arm, corpus, seed in models:
        if corpus == "play-v1":
            model, info = gr.load_v1_model(seed, "cuda", arm=arm)
        else:
            path = gr.model_dir(ROOT, arm, corpus, seed, debug=debug)
            model, info = gr.load_new_model(path, "cuda", feature_reports=pins, debug=debug)
        p = gm.predict_at(model, start, commands, states, horizons, chunk=1024, device="cuda")
        q = gm.predict_at(
            model, start, commands[wrong], states, horizons, chunk=1024, device="cuda"
        )
        e = {}
        for h in horizons:
            e[h] = {
                "true": gw.latent_error(p[h], targets[h]),
                "wrong": gw.latent_error(q[h], targets[h]),
                "copy": gw.latent_error(start, targets[h]),
            }
        name = f"{arm}-{gc.CORPUS_TAG.get(corpus, 'v1')}-{seed}"
        errors[name] = e
        results[name] = {"model": info, "stats": {}}
        for h in horizons:
            results[name]["stats"][str(h)] = {
                "copy_ratio": gw.ratio_stat(e[h]["true"], e[h]["copy"], weights, gc.E_LEVEL),
                "sens": gw.ratio_stat(e[h]["wrong"], e[h]["true"], weights, gc.E_LEVEL),
                "mean_true_error": float(e[h]["true"].mean()),
                "mean_true_error_lift_windows": float(e[h]["true"][classes["lift"]].mean())
                if classes["lift"].any()
                else None,
                "mean_true_error_contact_windows": float(e[h]["true"][classes["contact"]].mean())
                if classes["contact"].any()
                else None,
            }
        log(
            f"{name}: "
            + json.dumps(
                {h: results[name]["stats"][str(h)]["copy_ratio"]["value"] for h in horizons}
            )
        )
        del model
    checks, ok = {}, True
    seeds = (gc.DEBUG_MODEL_SEEDS[0],) if debug else gc.MODEL_SEEDS
    for corpus in gc.CORPORA:
        tag = gc.CORPUS_TAG[corpus]
        n_seed = seeds[0] if debug else gc.N_SEED
        for seed in seeds:
            name = f"G-{tag}-{seed}"
            vs_n = None
            if seed == n_seed:
                n = errors[f"N-{tag}-{n_seed}"]
                vs_n = {
                    str(h): {
                        "acc_ratio": gw.ratio_stat(
                            errors[name][h]["true"], n[h]["true"], weights, gc.E_LEVEL
                        )
                    }
                    for h in horizons
                }
            checks[name] = gc.e_checks(results[name]["stats"], vs_n)
            ok = ok and checks[name]["pass"]
    # reported: R25.18's lambda on the common val store with the pinned moments
    buffers = gr.pinned_buffers()
    report["lambda_reported"] = zs.pose_lambda(
        val.latents, val.state, val.table, buffers["state_mean"], buffers["state_scale"]
    )
    report |= {
        "results": results,
        "checks": checks,
        "row": "E-PASS" if ok else "E-FAIL",
        "level": gc.E_LEVEL,
        "finished_utc": utc(),
    }
    write_json(args.output, report)
    log(f"Stage E: {report['row']}")
    return 0


# ----- A-K0 and Stage A′: roots and costs (TASK-089's Part A) ------------------------------------
def _roots_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zg.build_root(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_roots(args) -> int:
    if args.cohort not in gc.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"roots-{args.cohort}.json", out / f"roots-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = list(gc.COHORTS[args.cohort])[: args.limit or None]
    if args.limit and not args.cohort.startswith("debug"):
        raise rt.GuardError("--limit is for debug cohorts only")
    t0 = time.time()
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_roots_worker, [c for c in chunks if c])
    roots = sorted((r for p in parts for r in p), key=lambda r: r["seed"])
    arrays, meta = {}, []
    for r in roots:
        s = r["seed"]
        row = {k: r[k] for k in ("seed", "attempt", "layout", "target", "yaw", "z0", "rejected")}
        for k, g in enumerate(r["subgoals"]):
            arrays[f"{s}/subgoal{k}/frame"] = g["frame"]
            arrays[f"{s}/subgoal{k}/state28"] = g["state28"]
        for root, rec in r["roots"].items():
            arrays[f"{s}/{root}/frame"] = rec["frame"]
            arrays[f"{s}/{root}/state28"] = rec["state28"]
        row["branches"] = {}
        for key, b in r["branches"].items():
            for k in ("actions", "frames", "states"):
                arrays[f"{s}/{key}/{k}"] = b[k]
            row["branches"][key] = {
                "rise": b["rise"],
                "contact": b["contact"],
                "max_rise": float(max(b["rise"])),
            }
        meta.append(row)
    np.savez_compressed(npath, **arrays)
    report |= {
        "seconds": time.time() - t0,
        "roots": meta,
        "arrays_sha256": sha(npath),
        "rejections": sum(len(r["rejected"]) for r in roots),
        "finished_utc": utc(),
    }
    write_json(jpath, report)
    log(f"{len(roots)} roots, {report['rejections']} rejected draws, {report['seconds']:.0f} s")
    return 0


def load_roots(folder, cohort, revision):
    folder = Path(folder)
    rep = json.loads((folder / f"roots-{cohort}.json").read_text())
    npath = folder / f"roots-{cohort}.npz"
    if sha(npath) != rep["arrays_sha256"]:
        raise rt.GuardError("G-roots: arrays differ from the roots report")
    seeds = [r["seed"] for r in rep["roots"]]
    for s in seeds:
        gc.check_seed(cohort, s)
    if not cohort.startswith("debug"):
        if seeds != list(gc.COHORTS[cohort]):
            raise rt.GuardError("G-roots: the cohort's seeds are not exactly its roots")
        if rep["revision"] != revision:
            raise rt.GuardError("G-revision: the roots were made at another revision")
    return rep, np.load(npath), sha(folder / f"roots-{cohort}.json")


def _encoded(enc_fn, data, rep):
    enc = {}
    for r in rep["roots"]:
        s = r["seed"]
        for root in zg.ROOTS:
            enc[f"{s}/{root}"] = enc_fn(data[f"{s}/{root}/frame"])[0]
            enc[f"{s}/goal/{root}"] = enc_fn(data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/frame"])[0]
            for name in zg.BRANCHES[root]:
                enc[f"{s}/{root}/{name}"] = enc_fn(data[f"{s}/{root}/{name}/frames"])
    return enc


def _ceiling(pm_by_family, enc, data, rep) -> dict:
    """{family: {h: {comparison: [win per root]}}} with the simulator as predictor (TASK-089)."""
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    out = {}
    for family, pm in pm_by_family.items():
        out[family] = {
            str(h): {zg.comparison_name(c): [] for c in zg.COMPARISONS} for h in zg.HORIZONS
        }
        for r in rep["roots"]:
            s = r["seed"]
            for comp in zg.COMPARISONS:
                root, t, a = comp
                gl, gs = enc[f"{s}/goal/{root}"], data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"]
                costs = {
                    b: zgr.ceiling_costs(
                        pm, enc[f"{s}/{root}/{b}"], data[f"{s}/{root}/{b}/states"], gl, gs
                    )
                    for b in (t, a)
                }
                for h in zg.HORIZONS:
                    w = bool(zg.wins(costs[t][h - 1], costs[a][h - 1]))
                    out[family][str(h)][zg.comparison_name(comp)].append(w)
    return out


def _pin_model_pms():
    """The ceiling's cost: G-87100's buffers give the state standardisation, and they are the
    pinned ``play-v1`` moments (checked bitwise); the model's weights are not used."""
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    model, info = gr.load_v1_model(87100, "cuda")
    return {
        "latent": gr.planner_model(model, w_pose=0.0),
        "latent+pose": gr.planner_model(model, w_pose=1.0),
    }, info


def cmd_ceiling(args) -> int:
    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    report = base_report(args.cohort)
    out = Path(args.output) / f"ceiling-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    devices.configure_determinism("cuda", strict=False)
    rep, data, roots_sha = load_roots(args.roots, args.cohort, report["revision"])
    pms, info = _pin_model_pms()
    report["moments_sha256"] = info["moments_sha256"]
    enc = _encoded(gr.encoder("cuda"), data, rep)
    wins = _ceiling(pms, enc, data, rep)
    rates = {
        f: {h: {c: float(np.mean(v)) for c, v in by_c.items()} for h, by_c in by_h.items()}
        for f, by_h in wins.items()
    }
    bars = zg.k0_bars(rates)
    report |= {
        "roots_sha256": roots_sha,
        "n": len(rep["roots"]),
        "wins": wins,
        "rates": rates,
        "bars": bars,
        "h8_qualifies": zg.qualifying(bars, 8),
        "h16_qualifies": zg.qualifying(bars, 16),
        "row": "A-CEILING" if not zg.qualifying(bars, 8) else "A-K0-PASS",
        "targets": {t: sum(r["target"] == t for r in rep["roots"]) for t in gc.pc.OBJECT_NAMES},
        "rejections": sum(len(r["rejected"]) for r in rep["roots"]),
        "finished_utc": utc(),
    }
    write_json(out, report)
    log(json.dumps(rates) + f" row {report['row']}")
    return 0


def shuffled_deal(seed: int, comparison: int, true_actions, alt_actions):
    """TASK-089's A1 shuffled control with A′'s salt 9002."""
    pool = np.concatenate([np.asarray(true_actions), np.asarray(alt_actions)])
    rng = np.random.default_rng([gc.SALTS["shuffle"], int(seed), int(comparison)])
    order = rng.permutation(len(pool))
    return pool[order[: zg.WINDOW]], pool[order[zg.WINDOW :]]


def _a_prime_models(debug):
    seeds = (gc.DEBUG_MODEL_SEEDS[0],) if debug else gc.MODEL_SEEDS
    out = [("G", c, s) for c in gc.CORPORA for s in seeds]
    out += [("N", c, seeds[0] if debug else gc.N_SEED) for c in gc.CORPORA]
    out += [("G", "play-v1", s) for s in (87100, 87101, 87102)]
    return out


def cmd_a_prime(args) -> int:
    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    debug = args.cohort.startswith("debug")
    report = base_report(args.cohort)
    out = Path(args.output) / f"a-prime-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    k0 = json.loads(Path(args.k0).read_text())
    if k0.get("row") != "A-K0-PASS" or k0.get("cohort") != "k0-roots" or k0["tracked_tree_dirty"]:
        raise rt.GuardError("G-k0: A′ needs A-K0's clean k0-roots ceiling (A-K0-PASS)")
    bars = k0["bars"]
    if gc.A1_BARS is not None and bars != gc.A1_BARS:
        raise rt.GuardError("G-k0: the ceiling file's bars are not the frozen ones")
    report["k0_sha256"] = sha(args.k0)
    devices.configure_determinism("cuda", strict=False)
    rep, data, roots_sha = load_roots(args.roots, args.cohort, report["revision"])
    enc = _encoded(gr.encoder("cuda"), data, rep)
    pins = _feature_pins(dict(zip(gc.CORPORA, args.debug_features, strict=True)) if debug else None)
    report["feature_reports_sha256"] = pins
    pms, _ = _pin_model_pms()
    report["ceiling"] = _ceiling(pms, enc, data, rep)
    seeds_r = [r["seed"] for r in rep["roots"]]
    folds = gc.probe_folds(np.repeat(seeds_r, 5))
    branches = [("R1", "close"), ("R1", "open"), ("R2", "lift"), ("R2", "release"), ("R2", "stay")]
    labels, ceil_x, floor_x = {}, {}, {}
    pin_model = pms["latent+pose"].model
    for h in zg.HORIZONS:
        y, cx, fx = [], [], []
        for r in rep["roots"]:
            s = r["seed"]
            for root, name in branches:
                b = r["branches"][f"{root}/{name}"]
                y.append(bool(b["rise"][h - 1] >= gc.LIFT_RISE and b["contact"][h - 1]))
                st = gr.norm_state(pin_model, data[f"{s}/{root}/{name}/states"][h - 1][None])[0]
                cx.append(np.r_[enc[f"{s}/{root}/{name}"][h - 1].ravel(), st])
                fx.append(st)
        labels[h], ceil_x[h], floor_x[h] = np.array(y), np.array(cx), np.array(fx)
    probe = {}
    for h in zg.HORIZONS:
        roots_of = np.repeat(seeds_r, len(branches))
        probe[str(h)] = {
            "ceiling": gc.auroc_interval(
                gc.probe_scores(ceil_x[h], labels[h], folds), labels[h], roots_of
            ),
            "proprio_floor": gc.auroc_interval(
                gc.probe_scores(floor_x[h], labels[h], folds), labels[h], roots_of
            ),
        }
    family_bars = bars["latent+pose"][str(gc.A_PRIME_H)]
    results = {}
    for arm, corpus, seed in _a_prime_models(debug):
        if corpus == "play-v1":
            model, info = gr.load_v1_model(seed, "cuda", arm=arm)
        else:
            path = gr.model_dir(ROOT, arm, corpus, seed, debug=debug)
            model, info = gr.load_new_model(path, "cuda", feature_reports=pins, debug=debug)
        name = f"{arm}-{gc.CORPUS_TAG.get(corpus, 'v1')}-{seed}"
        fams = {"latent": gr.planner_model(model, w_pose=0.0)}
        if arm == "G":
            fams["latent+pose"] = gr.planner_model(model, w_pose=1.0)
        res = {"model": info}
        for fam, pm in fams.items():
            w = {
                k: {
                    str(h): {zg.comparison_name(c): [] for c in zg.COMPARISONS} for h in zg.HORIZONS
                }
                for k in ("model", "shuffled")
            }
            for r in rep["roots"]:
                s = r["seed"]
                for ci, comp in enumerate(zg.COMPARISONS):
                    root, t, a = comp
                    rl, rs = enc[f"{s}/{root}"], data[f"{s}/{root}/state28"]
                    gl = enc[f"{s}/goal/{root}"]
                    gs = data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"]
                    ta, aa = data[f"{s}/{root}/{t}/actions"], data[f"{s}/{root}/{a}/actions"]
                    d1, d2 = shuffled_deal(s, ci, ta, aa)
                    costs = zgr.rollout_costs(pm, rl, rs, gl, gs, np.stack([ta, aa, d1, d2]))
                    c = zg.comparison_name(comp)
                    for h in zg.HORIZONS:
                        w["model"][str(h)][c].append(
                            bool(zg.wins(costs[0, h - 1], costs[1, h - 1]))
                        )
                        w["shuffled"][str(h)][c].append(
                            bool(zg.wins(costs[2, h - 1], costs[3, h - 1]))
                        )
            res[fam] = w
        if arm == "G":
            n = len(rep["roots"])
            res["a1_h8"] = zg.a1_pass(
                family_bars,
                res["latent+pose"]["model"][str(gc.A_PRIME_H)],
                res["latent+pose"]["shuffled"][str(gc.A_PRIME_H)],
                n,
            )
            firsts = {root: [] for root in zg.ROOTS}
            pm = fams["latent+pose"]
            for r in rep["roots"]:
                s = r["seed"]
                for root in zg.ROOTS:
                    firsts[root].append(
                        zgr.first_planned(
                            pm,
                            gc.A_PRIME_SETTING,
                            s,
                            enc[f"{s}/{root}"],
                            data[f"{s}/{root}/state28"],
                            enc[f"{s}/goal/{root}"],
                            data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"],
                        ).tolist()
                    )
            m = zg.a2_measures(firsts["R1"], firsts["R2"])
            res["a2"] = gc.a2_prime(m) | {"first_R1": firsts["R1"], "first_R2": firsts["R2"]}
            res["a_prime_pass"] = gc.a_prime_pass(res["a1_h8"], res["a2"])
            # the readability probe on this model's predictions (reported)
            res["probe"] = {}
            for h in zg.HORIZONS:
                px = []
                for r in rep["roots"]:
                    s = r["seed"]
                    for root, bname in branches:
                        pz, ps = gr.rollout_numpy(
                            model,
                            enc[f"{s}/{root}"][None],
                            data[f"{s}/{root}/{bname}/actions"][None],
                            data[f"{s}/{root}/state28"][None],
                        )
                        px.append(np.r_[pz[0, h - 1].ravel(), ps[0, h - 1]])
                px = np.array(px)
                roots_of = np.repeat(seeds_r, len(branches))
                res["probe"][str(h)] = gc.auroc_interval(
                    gc.probe_scores(px, labels[h], folds), labels[h], roots_of
                )
            log(
                f"{name}: A1 h8 pass {res['a1_h8']['pass']}, A2 "
                f"{res['a2']['close_R1']['count']}/{res['a2']['lift_R2']['count']}, "
                f"A′ {res['a_prime_pass']}"
            )
        results[name] = res
        del model
    passes = {
        tag: {
            s: results[f"G-{tag}-{s}"]["a_prime_pass"]
            for s in ((gc.DEBUG_MODEL_SEEDS[0],) if debug else gc.MODEL_SEEDS)
        }
        for tag in ("v2g", "v1p")
    }
    report |= {
        "roots_sha256": roots_sha,
        "n": len(rep["roots"]),
        "results": results,
        "probe": probe,
        "probe_positives": {str(h): int(labels[h].sum()) for h in zg.HORIZONS},
        "lift_branch_rise_ge5_16": sum(
            max(r["branches"]["R2/lift"]["rise"]) >= gc.REPORT_RISE for r in rep["roots"]
        ),
        "a_prime": gc.a_prime_row(passes),
        "finished_utc": utc(),
    }
    write_json(out, report)
    log(f"Stage A′: {report['a_prime']}")
    return 0


# ----- §7: goals, runs, video --------------------------------------------------------------------
def _goal_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zs.make_grasp_goal(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_goals(args) -> int:
    if args.cohort not in gc.GRASP_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a closed-loop cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"goals-{args.cohort}.json", out / f"goals-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = list(gc.COHORTS[args.cohort])[: args.limit or None]
    if args.limit and not args.cohort.startswith("debug"):
        raise rt.GuardError("--limit is for debug cohorts only")
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_goal_worker, [c for c in chunks if c])
    goals = sorted((g for p in parts for g in p), key=lambda g: g["seed"])
    arrays, meta = {}, []
    for g in goals:
        row = {k: v for k, v in g.items() if k != "subgoals"}
        row["subgoals"] = []
        for k, s in enumerate(g["subgoals"]):
            arrays[f"{g['seed']}_{k}_frame"] = s["frame"]
            row["subgoals"].append(
                {
                    "state28": s["state28"].tolist(),
                    "palm": s["palm"].tolist(),
                    "rotation": s["rotation"].tolist(),
                }
            )
        row["digest"] = zs.goal_digest(g)
        meta.append(row)
    np.savez_compressed(npath, **arrays)
    report |= {
        "seconds": time.time() - t0,
        "goals": meta,
        "frames_sha256": sha(npath),
        "rejections": sum(len(g["rejected"]) for g in goals),
        "finished_utc": utc(),
    }
    write_json(jpath, report)
    log(f"{len(goals)} goals, {report['rejections']} rejected draws, {report['seconds']:.0f} s")
    return 0


def load_goals(folder, cohort, revision=None):
    folder = Path(folder)
    rep = json.loads((folder / f"goals-{cohort}.json").read_text())
    npath = folder / f"goals-{cohort}.npz"
    if sha(npath) != rep["frames_sha256"]:
        raise rt.GuardError("G-goals: frames differ from the goal report")
    goals = []
    with np.load(npath) as frames:
        for row in rep["goals"]:
            gc.check_seed(cohort, row["seed"])
            g = dict(row)
            g["subgoals"] = [
                {
                    "frame": frames[f"{row['seed']}_{k}_frame"],
                    "state28": np.asarray(s["state28"], np.float32),
                    "palm": np.asarray(s["palm"], np.float32),
                    "rotation": np.asarray(s["rotation"], np.float32),
                }
                for k, s in enumerate(row["subgoals"])
            ]
            if zs.goal_digest(g) != row["digest"]:
                raise rt.GuardError(f"G-goals: seed {row['seed']} differs from its digest")
            goals.append(g)
    if not cohort.startswith("debug"):
        if [g["seed"] for g in goals] != list(gc.COHORTS[cohort]):
            raise rt.GuardError("G-goals: the cohort's seeds are not exactly its goals")
        if revision is not None and rep["revision"] != revision:
            raise rt.GuardError("G-goals: the goals were made at another revision")
    return goals, sha(folder / f"goals-{cohort}.json")


def check_run(cohort: str, arm: str, model_seed) -> None:
    """§7.1-§7.2: which (arm, model seed) a cohort admits."""
    if cohort not in gc.GRASP_COHORTS:
        raise rt.GuardError(f"{cohort} is not a closed-loop cohort")
    if arm in zs.BASELINE_ARMS:
        if model_seed is not None:
            raise rt.GuardError("baseline arms take no model seed")
        return
    if arm not in MODEL_ARMS:
        raise rt.GuardError(f"unknown arm {arm}")
    if cohort == "debug-grasp":
        return
    allowed = BD_SEEDS if cohort == "bd-grasp" else BS_SEEDS
    if model_seed not in allowed[arm]:
        raise rt.GuardError(f"{arm}-{model_seed} is not a run of {cohort}")


def _controller(arm, model_seed, episode_seed, cache, debug):
    if arm == "hold":
        return zs.HoldController()
    if arm == "random":
        return zs.RandomController(episode_seed)
    if arm == "ik":
        return zs.IKController()
    if arm == "ik-nopress":
        return zs.IKController(press=False)
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    if "model" not in cache:
        from embodied_jepa import devices

        devices.configure_determinism("cuda", strict=False)
        corpus = MODEL_ARMS[arm]
        if corpus == "play-v1":
            cache["model"], cache["info"] = gr.load_v1_model(model_seed, "cuda")
        else:
            path = gr.model_dir(ROOT, "G", corpus, model_seed, debug=debug)
            cache["model"], cache["info"] = gr.load_new_model(
                path, "cuda", feature_reports=cache["pins"], debug=debug
            )
        cache["encoder"] = gr.encoder("cuda")
    return gr.controller(arm, cache["model"], cache["encoder"], episode_seed)


def _run_worker(job):
    arm, model_seed, goals, pins, debug = job
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    cache, out = {"pins": pins}, []
    for goal in goals:
        controller = _controller(arm, model_seed, goal["seed"], cache, debug)
        result = zr.run_episode(robot, goal, controller)
        result["goal_digest"] = goal["digest"]
        if hasattr(controller, "setting"):
            result["setting"] = controller.setting
        out.append(result)
        log(
            f"{arm} {goal['seed']} success={result['success']} steps={result['steps']} "
            f"{result['stop_reason']}"
        )
    robot.close()
    return out, cache.get("info")


def cmd_run(args) -> int:
    report = base_report(args.cohort)
    debug = args.cohort.startswith("debug")
    check_run(args.cohort, args.arm, args.model_seed)
    name = f"{args.arm}-{args.model_seed}" if args.arm in MODEL_ARMS else args.arm
    goals, goals_sha = load_goals(args.goals, args.cohort, None if debug else report["revision"])
    if args.determinism:
        if (args.cohort, args.arm, args.model_seed) != ("bs-grasp", *DETERMINISM[:2]):
            raise rt.GuardError("the determinism re-run is M-v2g-90100 on bs-grasp (§7.2)")
        goals = goals[: DETERMINISM[2]]
        name += "-determinism"
    out = Path(args.output) / args.cohort / f"{name}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    if args.limit:
        if not debug:
            raise rt.GuardError("--limit is for debug cohorts only")
        goals = goals[: args.limit]
    pins = None
    if args.arm in MODEL_ARMS and MODEL_ARMS[args.arm] != "play-v1":
        pins = _feature_pins(
            dict(zip(gc.CORPORA, args.debug_features, strict=True)) if debug else None
        )
    report |= {
        "arm": args.arm,
        "model_seed": args.model_seed,
        "goals_sha256": goals_sha,
        "setting": "step-mean",
        "cem": zs.CEM,
        "lambda": zs.LAMBDA,
        "workers": args.workers,
        "feature_reports_sha256": pins,
        "determinism_rerun": bool(args.determinism),
    }
    t0 = time.time()
    workers = min(args.workers, len(goals))
    jobs = [(args.arm, args.model_seed, goals[i::workers], pins, debug) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_run_worker, [j for j in jobs if j[2]])
    episodes = sorted((e for p, _ in parts for e in p), key=lambda e: e["seed"])
    infos = [i for _, i in parts if i]
    report |= {
        "model": infos[0] if infos else None,
        "episodes": episodes,
        "successes": int(sum(e["success"] for e in episodes)),
        "n": len(episodes),
        "seconds": time.time() - t0,
        "finished_utc": utc(),
    }
    write_json(out, report)
    k, n, sec = report["successes"], report["n"], report["seconds"]
    log(f"{name} on {args.cohort}: {k}/{n} in {sec:.0f} s")
    return 0


def cmd_video(args) -> int:
    """Replay one episode's applied commands from its reset (checked against the run's palm
    distances) and write an MP4: the overview camera, the onboard frame and the current
    subgoal."""
    from PIL import Image, ImageDraw

    from embodied_jepa import play_corpus as pc

    base_report(gpu=True)
    result = json.loads(Path(args.result).read_text())
    goals, _ = load_goals(args.goals, result["cohort"])
    goal = next(g for g in goals if g["seed"] == args.seed)
    episode = next(e for e in result["episodes"] if e["seed"] == args.seed)
    robot = pc.make_play_robot()
    sim = robot.sim
    zs.reset(robot, goal["layout"])
    import mujoco

    big = mujoco.Renderer(sim.model, height=360, width=480)
    verdict = "SUCCESS" if episode["success"] else "FAILURE"
    name = result["arm"] + (f"-{result['model_seed']}" if result.get("model_seed") else "")
    label = f"TASK-090 grasp | arm {name} (step-mean) | reset {goal['seed']} | {verdict} (sim)"
    frames = []

    def snap(index, step):
        big.update_scene(sim.data, camera="overview")
        over = big.render()
        onboard = np.kron(sim.render(), np.ones((2, 2, 1), np.uint8)).astype(np.uint8)[:, :224]
        goal_img = np.kron(goal["subgoals"][index]["frame"], np.ones((2, 2, 1), np.uint8))
        side = np.concatenate([onboard[:180], goal_img[:180]], 0)
        canvas = np.zeros((400, 480 + 224, 3), np.uint8)
        canvas[40:400, :480] = over
        canvas[40:400, 480:] = side
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text((6, 4), label, fill=(255, 255, 255))
        draw.text(
            (6, 20),
            f"step {step}  onboard (top right), {zs.SUBGOALS[index]} subgoal image (bottom right)",
            fill=(200, 200, 200),
        )
        frames.append(np.asarray(image))

    switch = zs.SubgoalSwitch(goal["subgoals"])
    replayed, index = [], 0
    for k, action in enumerate(episode["applied"]):
        obs = robot.observe()
        p = zs.proprio(robot, obs)
        index = switch.update(p["palm"], p["state28"][7:14], p["rotation"])
        snap(index, k)
        res = robot.execute(np.asarray(action, np.float32))
        if res.status not in ("applied", "clipped"):
            log(f"replay stopped at {k}: {res.reason}")
            break
        palm, _ = robot.ee_pose("right")
        replayed.append(float(np.linalg.norm(palm - goal["subgoals"][index]["palm"])))
    snap(index, len(episode["applied"]))
    frames += [frames[-1]] * 20
    ok = np.allclose(replayed, episode["distances"][: len(replayed)], atol=1e-4)
    log(f"replay matches the run's palm distances: {ok}")
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
           f"{w}x{h}", "-r", "20", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           args.output]  # fmt: skip
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)
    meta = {
        "label": label,
        "result_file": args.result,
        "result_sha256": sha(args.result),
        "seed": args.seed,
        "replay_matches": bool(ok),
        "frames": len(frames),
    }
    Path(args.output + ".json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"wrote {args.output} ({len(frames)} frames)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--output", required=True)
    v.add_argument("--main-checkout", default="/home/huhn/develop/emai/open-embodied-jepa")
    v.add_argument(
        "--task087-evidence", default=str(Path.home() / "develop/emai/evidence/task087-val")
    )
    pw = sub.add_parser("power")
    pw.add_argument("--output", required=True)
    pw.add_argument("--sims", type=int, default=20_000)
    f = sub.add_parser("featurise")
    f.add_argument("--corpus", choices=gc.CORPORA)
    f.add_argument("--debug-corpus")
    f.add_argument("--output")
    f.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("stage-e")
    e.add_argument("--output", required=True)
    e.add_argument("--debug-features", nargs=2, metavar=("V2G", "V1P"))
    ro = sub.add_parser("roots")
    ro.add_argument("--cohort", required=True, choices=gc.ROOT_COHORTS)
    ro.add_argument("--output", required=True)
    ro.add_argument("--workers", type=int, default=8)
    ro.add_argument("--limit", type=int)
    ce = sub.add_parser("ceiling")
    ce.add_argument("--cohort", required=True, choices=gc.ROOT_COHORTS)
    ce.add_argument("--roots", required=True)
    ce.add_argument("--output", required=True)
    ap = sub.add_parser("a-prime")
    ap.add_argument("--cohort", required=True, choices=("a-roots", "debug-roots"))
    ap.add_argument("--roots", required=True)
    ap.add_argument("--k0", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--debug-features", nargs=2, metavar=("V2G", "V1P"))
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=gc.GRASP_COHORTS)
    g.add_argument("--output", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int)
    r = sub.add_parser("run")
    r.add_argument("--cohort", required=True, choices=gc.GRASP_COHORTS)
    r.add_argument("--arm", required=True, choices=sorted(MODEL_ARMS) + list(zs.BASELINE_ARMS))
    r.add_argument("--model-seed", type=int)
    r.add_argument("--goals", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--limit", type=int)
    r.add_argument("--determinism", action="store_true")
    r.add_argument("--debug-features", nargs=2, metavar=("V2G", "V1P"))
    vi = sub.add_parser("video")
    vi.add_argument("--result", required=True)
    vi.add_argument("--goals", required=True)
    vi.add_argument("--seed", type=int, required=True)
    vi.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return {
        "verify": cmd_verify,
        "power": cmd_power,
        "featurise": cmd_featurise,
        "stage-e": cmd_stage_e,
        "roots": cmd_roots,
        "ceiling": cmd_ceiling,
        "a-prime": cmd_a_prime,
        "goals": cmd_goals,
        "run": cmd_run,
        "video": cmd_video,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
