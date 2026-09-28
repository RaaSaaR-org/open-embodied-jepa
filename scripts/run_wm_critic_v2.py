"""TASK-073 runner (``apple_wm_critic_v2``): the staged run from K0 to the gated cohorts.

Protocol ``docs/experiments/apple_wm_critic_v2.md``; design ``src/embodied_jepa/wm_critic_v2.py``;
workers ``wm_critic_v2_runtime.py``; offline pieces ``wm_critic_v2_offline.py``. The corpus is
collected by ``scripts/collect_apple_shift_v2.py``, the ranking stage (O3, O4) runs in
``scripts/rank_wm_critic_v2.py`` and the offline decision and results manifests come from
``scripts/summarize_wm_critic_v2.py``.

Modes (every stage writes ``<output>/report.json``, refuses to overwrite it, and is V on any
guard, crash or cap):

- ``preflight`` -- the guards, G-evidence and G-repro (TASK-072 run-1's readouts refitted and
  reproduced exactly); simulates no TASK-073 seed.
- ``smoke`` -- mechanics on smoke seeds 53950-53999 only; nothing in it is read.
- ``k0`` -- the condition calibration on cohort K0 (development, no world model).
- ``train`` -- featurisation, readouts, blind baselines, the budget rule, W and N x 3, O1, O2.
- ``d3`` -- the development closed loop on cohort D3 (after OFFLINE-PASS).
- ``gated`` -- cohorts S and U; refused until an owner authorisation record is pinned in the
  manifest (``gated_authorization``), as for TASK-072's M2.

**Learned Apple->Plate on the frozen benchmark is still 0 successes.**
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2 as fp2  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import first_policy_v2_m2 as fm  # noqa: E402
from embodied_jepa import wm_critic_v2 as wc  # noqa: E402
from embodied_jepa import wm_critic_v2_runtime as rtm  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-wm-critic-v2.json"
fpl.configure_headless()  # MUJOCO_GL=egl before any MuJoCo import; workers inherit it
MODES = ("preflight", "smoke", "k0", "train", "d3", "gated")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M2R = _load("_task072_m2_runner", "scripts/run_first_policy_v2_m2.py")  # pinned, unmodified
R65 = M2R.R65
LIN = M2R.LIN
log = R65.log


# ----- pool ---------------------------------------------------------------------------------------
class Pool:
    def __init__(self, workers: int, config: dict):
        self.pool = mp.get_context("spawn").Pool(
            workers, initializer=rtm.worker_init, initargs=(config,)
        )
        self.look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        started = time.monotonic()
        result = self.pool.map_async(rtm.run_task, tasks, chunksize=1)
        try:
            out = result.get(timeout=max(cap - (time.monotonic() - started), 1.0))
        except mp.TimeoutError as error:
            raise wc.GuardError(f"G-cap: {what} exceeded its {cap} s cap") from error
        self.look_reference = LIN.check_look_states(out, self.look_reference)
        return out

    def close(self):
        self.pool.terminate()
        self.pool.join()


# ----- preflight ----------------------------------------------------------------------------------
def preflight(report: dict, mode: str, evidence: Path) -> dict:
    """G-frozen, G-hash, clean tree, G-platform, G-device, MuJoCo, G-weights, G-evidence, seeds."""
    import mujoco
    import torch

    from embodied_jepa import pretrained_encoder as pe

    manifest = json.loads(MANIFEST.read_text())
    if manifest["frozen"] != wc.frozen_block():
        raise wc.GuardError("G-frozen: the manifest's frozen block differs from the module")
    report["pinned_hashes_at_preflight"] = R65.check_pins(manifest["hashes"])
    dirty = R65.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
    if mode != "smoke":
        R65.check_clean(dirty)
    report["platform"] = fpl.check_platform()
    if not devices.available("cuda"):
        raise wc.GuardError("G-device: CUDA is not available")
    device = devices.require("cuda", strict=True)
    torch.set_num_threads(6)
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise wc.GuardError(f"G-hash: MuJoCo {mujoco.__version__} is not the pinned version")
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "accelerator": devices.accelerator_info(device),
        "determinism": devices.determinism_state(),
    }
    digests = {
        "pretrained": pe.weights_digest(pe.load_pretrained()),
        "floor": pe.weights_digest(pe.random_init()),
    }
    report["encoder_digests"] = digests
    if digests != manifest["encoder_digests"]:
        raise wc.GuardError("G-weights: an encoder digest differs from its pin")
    wc.check_seed_ranges()
    for role in wc.COHORT_ROLES:
        wc.stored_cohort(manifest, role)  # G-cohort: digests and whitelists (a JSON read)
    found = M2R.check_evidence(evidence)  # TASK-072 run-1's report, checkpoints, corpus
    report["evidence"] = {"root": str(evidence), "sha256": found["sha256"]}
    report["_run1"] = found["run1"]
    return manifest


def end_checks(report: dict, manifest: dict, mode: str) -> None:
    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])
    fpl.check_determinism(report)
    report["revision_at_end"] = R65.revision()
    if mode != "smoke" and R65.tracked_tree_dirty():
        raise wc.GuardError("G-hash: the tracked tree changed during the run")


def p3_checkpoint(evidence: Path) -> str:
    return str(evidence / wc.EVIDENCE["checkpoint_dir"] / f"{wc.CARRIED}.pt")


# ----- G-repro: TASK-072 run-1's readouts, refitted and reproduced exactly -----------------------
def refit_p_readout(report: dict, pool: Pool, evidence: Path, run1: dict):
    """M2's refit (``run_first_policy_v2_m2`` §2, G-repro): every recorded readout fact of
    run-1 must be reproduced exactly; returns the P readout and the encoder."""
    from embodied_jepa import first_policy_perception as fpp
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import pretrained_encoder as pe

    started = time.monotonic()
    pretrained, floor = pe.load_pretrained(), pe.random_init()
    reader = rt2.CorpusReader(
        evidence / fm.EVIDENCE["corpus"], fm.EVIDENCE["corpus_manifest_sha256"], splits=("train",)
    )
    train_ids = list(reader.manifest["splits"]["train"])
    if train_ids != run1["data"]["train_roots"]:
        raise wc.GuardError("G-data: the corpus's train roots are not run-1's")
    roots = []
    for episode_id in train_ids:
        arrays, meta = reader.episode(episode_id, keys=("phase",))
        roots.append(
            {
                "episode_id": episode_id,
                "success": bool(meta["success"]),
                "frame0": arrays["frame0"],
                "xy": meta["truth_xy"],
            }
        )
    wide_reset = LIN.load_wide_reset()
    frame_tasks = []
    for role in ("perception_train", "perception_heldout", "D"):
        for seed in M2R.check_rerender_seeds(role, run1["data"]["seeds"][role]):
            r = wide_reset(seed)
            reset = {
                "object_xy": list(map(float, r["object_xy"])),
                "plate_xy": list(map(float, r["plate_xy"])),
            }
            frame_tasks.append({"kind": "frame", "seed": seed, "reset": reset, "role": role})
    rendered = pool.map(frame_tasks, 1800.0, "re-render")
    frames = {role: [] for role in ("perception_train", "perception_heldout", "D")}
    for task, out in zip(frame_tasks, rendered, strict=True):
        frames[task["role"]].append(out)
    fit_frames = [r["frame0"] for r in roots] + [f["frame"] for f in frames["perception_train"]]
    fit_xy = np.asarray(
        [r["xy"] for r in roots] + [f["truth_xy"] for f in frames["perception_train"]]
    )
    readouts, feats = {}, {}
    for name, encoder in (("P", pretrained), ("R", floor)):
        feats[name] = fpp.featurise(encoder, np.stack(fit_frames))
        readouts[name] = fpp.XYReadout(feats[name], fit_xy)
    cross_p, _ = fpp.cross_fitted(feats["P"], fit_xy)
    c_mean = cross_p[: len(roots)].mean(axis=0)
    held = [f["frame"] for f in frames["perception_heldout"]]
    held_xy = np.asarray([f["truth_xy"] for f in frames["perception_heldout"]])
    apple_err, plate_err = fpp.errors_cm(
        readouts["P"].predict(fpp.featurise(pretrained, np.stack(held))), held_xy
    )
    library = [r for r in roots if r["success"]]
    lib_cls = fpp.featurise_cls(pretrained, [r["frame0"] for r in library])
    mu, sd = lib_cls.mean(axis=0), np.maximum(lib_cls.std(axis=0), fp2.INPUT_STD_FLOOR)
    cls = fpp.featurise_cls(pretrained, [f["frame"] for f in frames["D"]])
    d2_nearest = [
        int(np.argmin(np.linalg.norm((lib_cls - mu) / sd - (c - mu) / sd, axis=1))) for c in cls
    ]
    recomputed = {
        "selection": {n: json.loads(json.dumps(readouts[n].selection)) for n in readouts},
        "fit_rows": len(fit_xy),
        "apple_errors_cm": M2R._json_floats(apple_err),
        "plate_errors_cm": M2R._json_floats(plate_err),
        "c_mean": M2R._json_floats(c_mean),
        "library": [r["episode_id"] for r in library],
        "d2_nearest": [library[j]["episode_id"] for j in d2_nearest],
    }
    report["stages"]["reproduction"] = {
        "checks": M2R.check_reproduction(recomputed, run1),
        "seconds": time.monotonic() - started,
        "decoded_train_roots": len(reader.decoded),
        "test_split_decoded": reader.test_split_decoded,
    }
    return readouts["P"], pretrained


def cohort_estimates(pool: Pool, readout, encoder, seeds, resets, cap: float) -> dict:
    """Post-look frames of a cohort (stored resets) and P-3's post-look estimates from them."""
    from embodied_jepa import first_policy_perception as fpp

    frames = pool.map(
        [{"kind": "frame", "seed": s, "reset": resets[s]} for s in seeds], cap, "cohort frames"
    )
    for s, f in zip(seeds, frames, strict=True):
        if f["seed"] != s:
            raise wc.GuardError("G-cohort: a rendered frame is not its seed's")
    tokens = np.concatenate([fpp.featurise(encoder, f["frame"][None]) for f in frames])
    estimates = readout.predict(tokens)
    return {
        s: {
            "estimates": estimates[i].tolist(),
            "frame_sha256": frames[i]["post_look_frame_sha256"],
            "truth_xy": frames[i]["truth_xy"],
        }
        for i, s in enumerate(seeds)
    }


def arm_tasks(arm: str, seeds, resets, estimates, shift_of, **extra) -> list[dict]:
    return [
        {
            "kind": "attempt",
            "arm": arm,
            "seed": s,
            "reset": {"object_xy": resets[s]["object_xy"], "plate_xy": resets[s]["plate_xy"]},
            "shift": shift_of(s),
            "estimates": estimates[s]["estimates"],
            "expected_frame_sha256": estimates[s]["frame_sha256"],
        }
        | (extra.get("per_seed", lambda s: {})(s))
        | {k: v for k, v in extra.items() if k != "per_seed"}
        for s in seeds
    ]


def strip(records: list[dict]) -> list[dict]:
    """Per-attempt facts for the report (commands, latents and timings summarised)."""
    out = []
    for r in records:
        item = {
            k: v
            for k, v in r.items()
            if k not in ("commands", "latents", "act_seconds", "decisions")
        }
        decisions = r.get("decisions") or []
        item["decisions"] = [{k: v for k, v in d.items() if k != "costs_cm"} for d in decisions]
        item["decision_seconds"] = [d["seconds"] for d in decisions]
        out.append(item)
    return out


# ----- stage smokes: every stage's code on smoke seeds, with tiny budgets; nothing is read -------
SMOKE_ROLES = {"K0": (53950, 53953), "R": (53954, 53957), "D3": (53958, 53961)}
SMOKE_CORPUS = (53962, 53973)  # 8 train, 2 val, 2 test
SMOKE_BUDGET = {
    "calibration_updates": 200,
    "calibration_select_every": 100,
    "updates": 200,
    "select_every": 100,
}


def cohort(manifest: dict, role: str, smoke: bool):
    """The stored values of a cohort (G-cohort, G-seeds), or in a stage smoke stand-ins on
    smoke seeds generated by the same rules."""
    if not smoke:
        resets = wc.stored_cohort(manifest, role)
        return resets, wc.check_role_seeds(role, tuple(resets))
    low, high = SMOKE_ROLES[role]
    seeds = wc.check_role_seeds(role, tuple(range(low, high + 1)), smoke=True)
    resets = {}
    for s in seeds:
        r = wc.wide_reset_values(s)
        resets[s] = {
            "seed": s,
            **r,
            "shift_m": {str(m): wc.shift_vector(s, r, m) for m in wc.SHIFT_GRID_CM},
        }
    return resets, seeds


def read_stage_report(path, sha, outcome, smoke: bool) -> dict:
    """G-evidence for an earlier stage's report: its sha256, its outcome, and never a smoke
    report in a real stage (or the reverse)."""
    if R65.sha256_file(path) != sha:
        raise wc.GuardError(f"G-evidence: {path} differs from its sha256")
    report = json.loads(Path(path).read_text())
    if report.get("outcome") != outcome or bool(report.get("smoke")) != bool(smoke):
        raise wc.GuardError(f"G-evidence: {path} is not a recorded {outcome} run of this kind")
    return report


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


# ----- K0 -----------------------------------------------------------------------------------------
def stage_k0(report, manifest, evidence, clock, smoke=False):
    resets, seeds = cohort(manifest, "K0", smoke)
    pool = Pool(wc.SIM_WORKERS, {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": 1})
    report["_pool"] = pool
    readout, encoder = refit_p_readout(report, pool, evidence, report["_run1"])
    report["cohort_first_render_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    est = cohort_estimates(pool, readout, encoder, seeds, resets, 1800.0)
    cells = {}
    chosen = None
    o5 = None
    for step in wc.SHIFT_STEPS if not smoke else (300,):
        for cm in wc.SHIFT_GRID_CM if not smoke else (4,):

            def shift_of(s, step=step, cm=cm):
                return {"step": step, "vector": resets[s]["shift_m"][str(cm)]}

            counts, per_reset, records = {}, {}, {}
            while (
                arm := wc.k0_next_arm(counts)
                if not smoke
                else next((a for a in wc.K0_ARMS if a not in counts), None)
            ) is not None:
                clock.check(f"K0 {step}/{cm} {arm}")
                extra = {}
                if arm == "P-truth":
                    extra["log_chunks"] = True
                if arm == "H-sim":
                    extra["h_sim_centre"] = "truth"
                recs = pool.map(
                    arm_tasks(arm, seeds, resets, est, shift_of, **extra), 7200.0, f"K0 {arm}"
                )
                counts[arm] = sum(successes(recs))
                per_reset[arm] = successes(recs)
                records[arm] = recs
                log(f"K0 step {step} |d| {cm} cm: {arm} {counts[arm]}/32")
            verdict = wc.k0_cell_passes(counts)
            cells[f"{step}/{cm}"] = {
                "counts": counts,
                "per_reset": per_reset,
                **verdict,
                "attempts": {a: strip(r) for a, r in records.items()},
            }
            if verdict["passes"] or smoke:
                chosen = (step, cm)
                errors = [e for r in records["P-truth"] for e in r.get("o5_relative_errors", [])]
                o5 = wc.o5_passes(errors) | {"decisions": len(errors), "values": errors}
                break
        if chosen:
            break
    selection = wc.k0_select(
        {tuple(int(x) for x in k.split("/")): v["counts"] for k, v in cells.items()}
    )
    if smoke:  # the stage smoke forces a condition so later stage smokes can run
        selection = {"row": "K0-PASS", "shift_step": 300, "shift_cm": 4, "smoke_forced": True}
    return {
        "cells": cells,
        "selection": selection,
        "o5": o5,
        "estimates": {str(s): v for s, v in est.items()},
    }


# ----- training (offline): features, readouts, baselines, budget, W and N, O1, O2 ---------------
def stage_train(report, manifest, evidence, clock, args):
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa import token_dynamics as td
    from embodied_jepa import wm_critic_v2_offline as off
    from embodied_jepa.embodiment import G1Embodiment  # noqa: F401 - the schema's owner

    k0 = read_stage_report(args.k0_report, args.k0_sha256, "K0-PASS", args.smoke)
    shift_step = int(k0["stages"]["k0"]["selection"]["shift_step"])
    corpus = Path(args.corpus)
    reader = rt2.CorpusReader(corpus, args.corpus_sha256, splits=wc.READ_SPLITS)
    report["_reader"] = reader
    look = rt2.CorpusReader(
        evidence / fm.EVIDENCE["corpus"],
        fm.EVIDENCE["corpus_manifest_sha256"],
        splits=("train", "val"),
    )
    device = "cuda"
    encoder = pe.load_pretrained()
    band = wc.FEATURE_BAND
    table = off.Table()
    rmid_rows = {"train": [], "val": []}
    anchor_frames, anchor_rows = [], []
    started = time.monotonic()
    keys = (*rt2.EPISODE_ARRAYS, *wc.EXTRA_EPISODE_ARRAYS)
    sources = [("shift", reader, s) for s in wc.READ_SPLITS] + [
        ("look", look, s) for s in ("train", "val")
    ]
    for source, rd, split in sources:
        for episode_id in rd.manifest["splits"][split]:
            clock.check("features")
            arrays, meta = rd.episode(
                episode_id, keys=keys if source == "shift" else rt2.EPISODE_ARRAYS
            )
            frames = arrays["frames"]
            hi = min(band[1], len(frames) - 1)
            if hi <= band[0] + wc.CHUNK:
                continue  # the root ended before the band holds one window
            sl = slice(band[0], hi + 1)
            if source == "shift":
                plate = arrays["plate"][sl]
            else:
                plate = np.repeat(np.asarray([[*meta["truth_xy"][2:], 0.746]]), hi + 1 - band[0], 0)
            pooled = off.pooled_features(encoder, frames[sl], device=device)
            actions = np.zeros((len(pooled), 14), np.float32)
            n_act = min(len(arrays["applied"]) - band[0], len(pooled))
            actions[:n_act] = arrays["applied"][band[0] : band[0] + n_act]
            root = {
                "id": episode_id,
                "split": split if source == "shift" else f"look-{split}",
                "source": source,
                "shift_step": (meta.get("shift") or {}).get("step") if source == "shift" else None,
            }
            table.add(root, pooled, actions, arrays["apple"][sl], plate, band[0])
            if source == "shift" and len(anchor_frames) < wc.FEATURE_ANCHOR["frames"]:
                take = min(16, wc.FEATURE_ANCHOR["frames"] - len(anchor_frames))
                anchor_frames.extend(frames[sl][:take])
                anchor_rows.extend(range(table.size - len(pooled), table.size - len(pooled) + take))
            if source == "shift":
                for t in wc.DECISION_STEPS:
                    if t < len(frames):
                        rmid_rows[split].append((episode_id, frames[t], arrays["plate"][t, :2]))
    table.seal()
    report["stages"]["features"] = {
        "frames": int(table.size),
        "roots": len(table.roots),
        "seconds": time.monotonic() - started,
        "anchor": off.anchor_check(
            encoder, np.stack(anchor_frames), table.features[np.asarray(anchor_rows)]
        ),
        "test_split_decoded": reader.test_split_decoded,
    }
    # R-mid on full tokens (CPU, batch size 1: the closed loop's path)
    started = time.monotonic()
    tok = {
        s: off.full_tokens_cpu(encoder, np.stack([r[1] for r in rmid_rows[s]]))
        for s in ("train", "val")
    }
    r_mid = off.fit_r_mid(
        tok["train"],
        np.stack([r[2] for r in rmid_rows["train"]]),
        [r[0] for r in rmid_rows["train"]],
    )
    val_err = 100 * np.linalg.norm(
        r_mid.predict(tok["val"]) - np.stack([r[2] for r in rmid_rows["val"]]), axis=1
    )
    del tok
    idx = {
        s: [i for i, r in enumerate(table.roots) if r["split"] == s]
        for s in ("train", "val", "look-train", "look-val")
    }
    r_off, r_off_rows = off.fit_r_off(table, idx["train"])
    out_dir = Path(args.output)
    np.savez(out_dir / "r_mid.npz", **r_mid.state())
    np.savez(out_dir / "r_off.npz", **r_off.state())
    report["stages"]["readouts"] = {
        "r_mid": r_mid.selection
        | {
            "sha256": r_mid.sha256(),
            "val_error_cm_median": float(np.median(val_err)),
            "val_error_cm_p90": float(np.quantile(val_err, 0.9)),
        },
        "r_off": r_off.selection | {"sha256": r_off.sha256()},
        "seconds": time.monotonic() - started,
    }
    # blind baselines, recorded before any W number exists
    o2_starts, o2_owners = table.windows(
        idx["val"],
        start_steps={t for t in wc.DECISION_STEPS if t >= max(wc.O2["first_step"], shift_step)},
    )
    report["stages"]["blind_baselines"] = off.o2_statistics(
        {}, {}, r_off, table, o2_starts, o2_owners
    )["baselines"]
    report["stages"]["blind_baselines"]["copy_last_o2"] = "ratio 1.0 by definition (persistence)"
    R65.write_report(out_dir / "baselines.json", report["stages"]["blind_baselines"])
    # training context
    train_roots = idx["train"] + idx["look-train"] + idx["look-val"]
    train_starts, _ = table.windows(train_roots)
    val_starts, _ = table.windows(idx["val"], stride=4)
    rows = np.concatenate(
        [
            np.arange(table.roots[i]["first"], table.roots[i]["first"] + table.roots[i]["length"])
            for i in train_roots
        ]
    )
    mean, std = R65_moments(table.features, rows)
    scale = np.maximum(std, ld_floor())
    schema = rt2.make_robot().state_schema
    ctx = {
        "table": table,
        "train_starts": train_starts,
        "val_starts": val_starts,
        "scale": scale,
        "mean": mean,
        "std": std,
        "schema": schema,
        "device": device,
        "train_ids": sorted(table.roots[i]["id"] for i in train_roots),
        "metadata": {"corpus_manifest_sha256": args.corpus_sha256},
        "cap_seconds": wc.CAPS_SECONDS["per_model"],
    }
    ckpt = Path(args.checkpoints)
    ckpt.mkdir(parents=True, exist_ok=False)
    b = dict(wc.BUDGET) | (SMOKE_BUDGET if args.smoke else {})
    calibration = {}
    for arm, seed in b["calibration_runs"]:
        clock.check(f"calibration {arm}")
        _m, rec = off.train_model(
            ctx, arm, seed, b["calibration_updates"], b["calibration_select_every"]
        )
        calibration[f"{arm}-{seed}"] = rec | {
            "saturation_update": td.saturation_update(rec["val_curve"], b["saturation_tolerance"])
        }
    budget = wc.budget_updates([v["saturation_update"] for v in calibration.values()])
    report["stages"]["budget"] = {"calibration": calibration, "rule": budget}
    if budget["escalate"] and not args.smoke:
        return {"outcome": "ESCALATE-BUDGET"}
    updates = budget["updates"] if not args.smoke else b["updates"]
    for attempt in range(2):
        models, records = {}, {}
        every = updates // b["selection_points"] if not args.smoke else b["select_every"]
        for arm in ("W", "N"):
            for seed in wc.MODEL_SEEDS:
                clock.check(f"model {arm}-{seed}")
                path = ckpt / f"{arm}-{seed}-u{updates}.pt"
                model, rec = off.train_model(ctx, arm, seed, updates, every, path)
                rec["checkpoint"] = str(path)
                rec["checkpoint_sha256"] = R65.sha256_file(path)
                models[(arm, seed)], records[f"{arm}-{seed}"] = model, rec
        last_two = any(r["selected_in_last_two_points"] for r in records.values())
        report["stages"][f"models_u{updates}"] = records
        if not last_two or args.smoke:
            break
        if attempt == 1 or updates >= b["cap"]:
            return {"outcome": "ESCALATE-BUDGET-LAST-TWO"}
        updates = min(2 * updates, b["cap"])
    w = {s: models[("W", s)] for s in wc.MODEL_SEEDS}
    n = {s: models[("N", s)] for s in wc.MODEL_SEEDS}
    primary = min(wc.MODEL_SEEDS, key=lambda s: (records[f"W-{s}"]["selected_val_criterion"], s))
    # O1 and O2
    enc = td.Moments(np.zeros(table.features.shape[1]))
    for lo in range(0, len(rows), 8192):
        enc.add(table.features[rows[lo : lo + 8192]] / scale)
    basis = td.projection_basis(enc)
    del enc
    windows = {
        "overall": table.windows(idx["val"], stride=4),
        "post_shift": table.windows(
            [i for i in idx["val"] if table.roots[i]["shift_step"] is not None],
            stride=4,
            start_min=shift_step,
        ),
    }
    o1 = off.o1_statistics(w, n, table, windows, scale, basis)
    o2 = off.o2_statistics(w, n, r_off, table, o2_starts, o2_owners)
    return {
        "outcome": "TRAIN-COMPLETE",
        "primary_w_seed": primary,
        "o1": {
            str(s): v["gate"]
            | {"statistics": {f"{k[0]}@h{k[1]}": v2 for k, v2 in v["statistics"].items()}}
            for s, v in o1.items()
        },
        "o2": {str(s): o2[s] for s in wc.MODEL_SEEDS},
        "o2_baselines": o2["baselines"],
        "critic": {
            "r_off": str(out_dir / "r_off.npz"),
            "r_mid": str(out_dir / "r_mid.npz"),
            "models": {
                f"W{i}": records[f"W-{s}"]["checkpoint"]
                for i, s in enumerate([primary, *[x for x in wc.MODEL_SEEDS if x != primary]])
            }
            | {"N": records[f"N-{primary}"]["checkpoint"]},
        },
    }


def R65_moments(features, rows):
    mean = np.zeros(features.shape[1])
    for lo in range(0, len(rows), 8192):
        mean += features[rows[lo : lo + 8192]].astype(np.float64).sum(0)
    mean /= len(rows)
    sq = np.zeros(features.shape[1])
    for lo in range(0, len(rows), 8192):
        d = features[rows[lo : lo + 8192]].astype(np.float64) - mean
        sq += (d * d).sum(0)
    return mean, np.sqrt(sq / len(rows))


def ld_floor():
    from embodied_jepa import latent_dynamics as ld

    return ld.METRIC_FLOOR_STD


# ----- closed-loop stages (D3 and gated) ------------------------------------------------------
def closed_loop(
    report, manifest, evidence, clock, role, arms, critic_files, shift, workers, smoke=False
):
    """Run ``arms`` once per reset of ``role`` (P-reread first: H-shuf reads its latents)."""
    resets, seeds = cohort(manifest, role, smoke)
    config = {
        "p3_checkpoint": p3_checkpoint(evidence),
        "torch_threads": wc.H_WORKER_TORCH_THREADS,
    } | critic_files
    pool = Pool(workers, config)
    report["_pool"] = pool
    readout, encoder = refit_p_readout(report, pool, evidence, report["_run1"])
    report["cohort_first_render_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    est = cohort_estimates(pool, readout, encoder, seeds, resets, 1800.0)

    def shift_of(s):
        if shift is None:
            return None
        return {"step": shift[0], "vector": resets[s]["shift_m"][str(shift[1])]}

    order = ["P-reread", *[a for a in arms if a != "P-reread"]]
    records = {}
    for arm in order:
        clock.check(f"{role} {arm}")
        extra = {}
        if arm == "H-shuf":
            latents = [r["latents"] for r in records["P-reread"]]
            extra["per_seed"] = lambda s, lat=latents: {
                "foreign": dict(lat[(seeds.index(s) + 1) % len(seeds)])
            }
        records[arm] = pool.map(
            arm_tasks(arm, seeds, resets, est, shift_of, **extra), 7200.0, f"{role} {arm}"
        )
        if arm in wc.L1_ARMS and not all(r["privileged_ok"] for r in records[arm]):
            raise wc.GuardError(f"G-privileged: {arm} read simulator truth inside act()")
    return seeds, est, records


def stage_d3(report, manifest, evidence, clock, args):
    offline = json.loads(Path(args.offline).read_text())
    if R65.sha256_file(args.offline) != args.offline_sha256 or bool(offline.get("smoke")) != bool(
        args.smoke
    ):
        raise wc.GuardError("G-evidence: the offline decision differs from its sha256 or kind")
    if offline["row"] != "OFFLINE-PASS" and not args.smoke:
        raise wc.GuardError("G-evidence: the offline decision is not OFFLINE-PASS")
    shift = (int(offline["shift_step"]), int(offline["shift_cm"]))
    seeds, _est, records = closed_loop(
        report,
        manifest,
        evidence,
        clock,
        "D3",
        wc.D3_ARMS,
        offline["critic"],
        shift,
        wc.H_WORKERS,
        smoke=args.smoke,
    )
    counts = {arm: sum(successes(r)) for arm, r in records.items()}
    decision = wc.decide_d3(counts)
    return {
        "counts_of_16": counts,
        "decision": decision,
        "attempts": {a: strip(r) for a, r in records.items()},
    }


def stage_gated(report, manifest, evidence, clock, args):
    if not manifest.get("gated_authorization"):
        raise wc.GuardError(
            "G-authorisation: no owner authorisation record is pinned for the gated stage"
        )
    raise wc.GuardError("G-authorisation: the gated stage is written by its authorisation PR")


# ----- smoke (mechanics only; smoke seeds; nothing is read) -------------------------------------
SMOKE = {
    "render_seeds": 32,
    "render_repeats": 4,
    "proposal_seeds": 4,
    "corpus_seeds": 8,
    "train_updates": 500,
    "train_select_every": 250,
    "latency_seeds": 8,
    "shift_step": 300,
    "shift_cm": 4,
}


def gpu_snapshot() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def stage_smoke(report, manifest, evidence, clock, args):
    """Render check at 16 workers, the proposal generator, a small smoke corpus, two identical
    W trainings on strict CUDA, and decision latency with GR00T resident; smoke seeds only."""
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa import wm_critic_v2_offline as off

    out_dir = Path(args.output)
    all_seeds = wc.check_role_seeds("smoke", wc.smoke_seeds(), smoke=True)
    resets = {s: wc.wide_reset_values(s) for s in all_seeds}
    for s in all_seeds:
        resets[s]["shift_m"] = {str(m): wc.shift_vector(s, resets[s], m) for m in wc.SHIFT_GRID_CM}
    config = {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": 1}
    pool = Pool(wc.SIM_WORKERS, config)
    report["_pool"] = pool
    readout, encoder = refit_p_readout(report, pool, evidence, report["_run1"])

    def shift_of(s):
        return {"step": SMOKE["shift_step"], "vector": resets[s]["shift_m"][str(SMOKE["shift_cm"])]}

    # (1) render check: post-look frames, each seed rendered 4 times across 16 workers
    seeds = all_seeds[: SMOKE["render_seeds"]]
    tasks = [
        {"kind": "frame", "seed": s, "reset": resets[s]}
        for _ in range(SMOKE["render_repeats"])
        for s in seeds
    ]
    rendered = pool.map(tasks, 1800.0, "render check")
    shas: dict = {}
    for t, o in zip(tasks, rendered, strict=True):
        shas.setdefault(t["seed"], set()).add(o["post_look_frame_sha256"])
    report["stages"]["render_check"] = {
        "workers": wc.SIM_WORKERS,
        "renders": len(tasks),
        "seeds": len(seeds),
        "verdict": "IDENTICAL" if all(len(v) == 1 for v in shas.values()) else "DIFFERENT",
    }
    # (2) the proposal generator on P-truth attempts with the shift
    p_seeds = all_seeds[: SMOKE["proposal_seeds"]]
    est = cohort_estimates(pool, readout, encoder, p_seeds, resets, 600.0)
    recs = pool.map(
        arm_tasks("P-truth", p_seeds, resets, est, shift_of, log_chunks=True),
        1800.0,
        "proposal smoke",
    )
    errors = [e for r in recs for e in r.get("o5_relative_errors", [])]
    report["stages"]["proposal_generator"] = {
        "attempts": len(recs),
        "decisions": len(errors),
        "shift_applied": [bool(r["shift_log"]) for r in recs],
        "relative_chunk_error_median": float(np.median(errors)) if errors else None,
        "note": "mechanics only on smoke seeds; not read for any decision or threshold",
    }
    # (3) a small smoke corpus (8 roots), GPU features, two identical W trainings
    c_seeds = all_seeds[SMOKE["render_seeds"] : SMOKE["render_seeds"] + SMOKE["corpus_seeds"]]
    folder = out_dir / "smoke-corpus"
    folder.mkdir(parents=True)
    ctasks = []
    for i, s in enumerate(c_seeds):
        ctasks.append(
            {
                "kind": "collect",
                "seed": s,
                "episode_id": f"smoke-{s}",
                "session_id": f"smoke-reset-{s}",
                "split": "train" if i < 6 else "val",
                "reset": {k: resets[s][k] for k in ("object_xy", "plate_xy")},
                "shift": shift_of(s) if i % 4 != 3 else None,
                "misaimed": i % 2 == 1,
                "misaim_m": [0.01, -0.01] if i % 2 == 1 else [0.0, 0.0],
                "noise_level": i % 4,
                "noise_seed": 1000 + i,
                "folder": str(folder),
            }
        )
    collected = pool.map(ctasks, 1800.0, "smoke corpus")
    pool.close()
    report.pop("_pool")
    table = off.Table()
    enc = pe.load_pretrained()
    rmid = []
    for t in ctasks:
        with np.load(folder / f"{t['episode_id']}.npz") as data:
            arrays = {k: data[k] for k in data.files}
        band = wc.FEATURE_BAND
        hi = min(band[1], len(arrays["frames"]) - 1)
        sl = slice(band[0], hi + 1)
        pooled = off.pooled_features(enc, arrays["frames"][sl], device="cuda")
        actions = np.zeros((len(pooled), 14), np.float32)
        n_act = min(len(arrays["applied"]) - band[0], len(pooled))
        actions[:n_act] = arrays["applied"][band[0] : band[0] + n_act]
        table.add(
            {
                "id": t["episode_id"],
                "split": t["split"],
                "source": "smoke",
                "shift_step": (t["shift"] or {}).get("step"),
            },
            pooled,
            actions,
            arrays["apple"][sl],
            arrays["plate"][sl],
            band[0],
        )
        for step in wc.DECISION_STEPS[::3]:
            if step < len(arrays["frames"]):
                rmid.append((t["episode_id"], arrays["frames"][step], arrays["plate"][step, :2]))
    table.seal()
    anchor_frames = []
    for t in ctasks[:2]:
        with np.load(folder / f"{t['episode_id']}.npz") as data:
            anchor_frames.extend(data["frames"][wc.FEATURE_BAND[0] : wc.FEATURE_BAND[0] + 16])
    anchor_rows = np.concatenate([np.arange(r["first"], r["first"] + 16) for r in table.roots[:2]])
    try:
        anchor = off.anchor_check(enc, np.stack(anchor_frames), table.features[anchor_rows])
    except wc.GuardError as error:  # recorded, not raised: the smoke measures it
        anchor = {"failed": str(error)}
    report["stages"]["feature_anchor_smoke"] = anchor
    train_idx = [i for i, r in enumerate(table.roots) if r["split"] == "train"]
    val_idx = [i for i, r in enumerate(table.roots) if r["split"] == "val"]
    r_off, _rows = off.fit_r_off(table, train_idx)
    r_mid = off.fit_r_mid(
        off.full_tokens_cpu(enc, np.stack([r[1] for r in rmid])),
        np.stack([r[2] for r in rmid]),
        [r[0] for r in rmid],
    )
    np.savez(out_dir / "r_off.npz", **r_off.state())
    np.savez(out_dir / "r_mid.npz", **r_mid.state())
    train_starts, _ = table.windows(train_idx)
    val_starts, _ = table.windows(val_idx, stride=4)
    rows = np.concatenate(
        [
            np.arange(table.roots[i]["first"], table.roots[i]["first"] + table.roots[i]["length"])
            for i in train_idx
        ]
    )
    mean, std = R65_moments(table.features, rows)
    ctx = {
        "table": table,
        "train_starts": train_starts,
        "val_starts": val_starts,
        "scale": np.maximum(std, ld_floor()),
        "mean": mean,
        "std": std,
        "schema": rt2.make_robot().state_schema,
        "device": "cuda",
        "train_ids": sorted(table.roots[i]["id"] for i in train_idx),
        "metadata": {"smoke": True},
        "cap_seconds": wc.CAPS_SECONDS["per_model"],
    }
    runs = []
    for k in range(2):
        path = out_dir / f"W-smoke-{k}.pt"
        model, rec = off.train_model(
            ctx, "W", wc.MODEL_SEEDS[0], SMOKE["train_updates"], SMOKE["train_select_every"], path
        )
        runs.append(
            {
                "weights_sha256": off.weights_sha256(model),
                "losses": rec["losses"],
                "val_curve": rec["val_curve"],
                "seconds": rec["seconds"],
                "checkpoint": str(path),
            }
        )
        del model
    identical = (
        runs[0]["weights_sha256"] == runs[1]["weights_sha256"]
        and runs[0]["losses"] == runs[1]["losses"]
        and runs[0]["val_curve"] == runs[1]["val_curve"]
    )
    report["stages"]["training_smoke"] = {
        "collected": [
            {k: c[k] for k in ("episode_id", "complete", "success", "shift_applied")}
            for c in collected
        ],
        "frames": int(table.size),
        "updates": SMOKE["train_updates"],
        "runs": runs,
        "verdict": "BIT-IDENTICAL" if identical else "DIFFERENT",
        "determinism": devices.determinism_state(),
        "cuda_memory": devices.memory_report("cuda"),
    }
    # (4) decision latency with the GPU's other resident service, H-LeWM on 8 workers x 2 threads
    snapshot_before = gpu_snapshot()
    critic_files = {
        "r_off": str(out_dir / "r_off.npz"),
        "r_mid": str(out_dir / "r_mid.npz"),
        "models": {"W0": runs[0]["checkpoint"], "N": runs[0]["checkpoint"]},
    }
    lpool = Pool(
        wc.H_WORKERS,
        {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": wc.H_WORKER_TORCH_THREADS}
        | critic_files,
    )
    report["_pool"] = lpool
    l_seeds = all_seeds[-SMOKE["latency_seeds"] :]
    l_est = cohort_estimates(lpool, readout, encoder, l_seeds, resets, 600.0)
    lrecs = lpool.map(
        arm_tasks("H-LeWM", l_seeds, resets, l_est, shift_of), 3600.0, "latency smoke"
    )
    seconds = [d["seconds"] for r in lrecs for d in r["decisions"]]
    report["stages"]["latency"] = {
        "workers": wc.H_WORKERS,
        "torch_threads": wc.H_WORKER_TORCH_THREADS,
        "decisions": len(seconds),
        "median_seconds": float(np.median(seconds)) if seconds else None,
        "p90_seconds": float(np.quantile(seconds, 0.9)) if seconds else None,
        "max_seconds": float(np.max(seconds)) if seconds else None,
        "gpu_before": snapshot_before,
        "gpu_after": gpu_snapshot(),
        "privileged_ok": [bool(r["privileged_ok"]) for r in lrecs],
        "note": "mechanics only on smoke seeds with a 500-update smoke W; not read",
    }
    return {"outcome": "SMOKE-COMPLETE"}


# ----- run ----------------------------------------------------------------------------------------
def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    evidence = Path(args.evidence).resolve()
    report = {
        "protocol": wc.PROTOCOL,
        "task": wc.TASK,
        "mode": args.mode,
        "smoke": bool(args.smoke or args.mode == "smoke"),
        "outcome": None,
        "stages": {},
        "paths": {"output": str(output), "evidence": str(evidence)},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(wc.CAPS_SECONDS["global"])
    try:
        kind = "smoke" if report["smoke"] else args.mode  # a stage smoke may run on a dirty tree
        manifest = preflight(report, kind, evidence)
        if args.mode == "preflight":
            pool = Pool(
                wc.SIM_WORKERS, {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": 1}
            )
            report["_pool"] = pool
            refit_p_readout(report, pool, evidence, report["_run1"])
            result = {"outcome": "PREFLIGHT-READY"}
        elif args.mode == "smoke":
            result = stage_smoke(report, manifest, evidence, clock, args)
        elif args.mode == "k0":
            k0 = stage_k0(report, manifest, evidence, clock, smoke=args.smoke)
            report["stages"]["k0_pending"] = True
            result = {"outcome": k0["selection"]["row"], "k0": k0}
        elif args.mode == "train":
            result = stage_train(report, manifest, evidence, clock, args)
        elif args.mode == "d3":
            d3 = stage_d3(report, manifest, evidence, clock, args)
            result = {"outcome": d3["decision"]["row"], "d3": d3}
        else:
            result = stage_gated(report, manifest, evidence, clock, args)
        end_checks(report, manifest, kind)
        report["stages"].pop("k0_pending", None)
        for key, value in result.items():
            if key == "outcome":
                report["outcome"] = value
            else:
                report["stages"][key] = value
    except Exception as error:  # noqa: BLE001 - every failure is V, and the report is written
        report["outcome"] = "V"
        report["void_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        log(f"VOID: {report['void_reason']}")
    finally:
        pool = report.pop("_pool", None)
        if pool is not None:
            pool.close()
        reader = report.pop("_reader", None)
        if reader is not None:
            report["decoded_episodes"] = len(reader.decoded)
            report["test_split_decoded"] = reader.test_split_decoded
        report.pop("_run1", None)
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_seconds"] = clock.elapsed()
        R65.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help="the checkout that holds TASK-072 run-1's outputs/, checkpoints/, data/",
    )
    parser.add_argument("--k0-report")
    parser.add_argument("--k0-sha256")
    parser.add_argument("--corpus")
    parser.add_argument("--corpus-sha256")
    parser.add_argument("--checkpoints")
    parser.add_argument("--offline")
    parser.add_argument("--offline-sha256")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="k0 / train / d3 on smoke seeds with tiny budgets (mechanics only; nothing is read)",
    )
    args = parser.parse_args(argv)
    if args.smoke and args.mode not in ("k0", "train", "d3"):
        parser.error("--smoke is for the k0, train and d3 stages (the smoke mode is its own)")
    need = {
        "train": ("k0_report", "k0_sha256", "corpus", "corpus_sha256", "checkpoints"),
        "d3": ("offline", "offline_sha256"),
    }.get(args.mode, ())
    missing = [n for n in need if getattr(args, n) is None]
    if missing:
        parser.error(f"{args.mode} needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
