"""TASK-100: roll-outs, readouts and statistics (protocol §6, §7, §9, §10.3-§10.6).

Passes (each a separate invocation, so the GPU lock is never held for CPU work):
  fitpred   (GPU) each model's roll-outs from every root of the fit view under its executed
            commands (N arms: zero), saved at every horizon for R-pred
  readouts  (CPU) R-enc (every fifth fit-view frame) and the copy-last readouts per encoder
            folder, and R-pred per model and horizon, all on the fit view
  evalpred  (GPU) each model's roll-outs from every root of the evaluation view under the true,
            wrong, zero, hold and candidate commands; latent errors, readout estimates and the
            true-command predictions
  score     (CPU) the statistics, the gates and (purpose ``test``) the row; ``calibrate`` derives
            Stage 0's bars, ``grid`` applies §10.3's rule from R-enc alone, ``val`` writes the val
            report, ``smoke`` exercises the row logic with val standing in for test
The fit view is D0-fit at Stage 0 and the fit sample of train at the gated stages; the evaluation
view is D0-eval (calibration), val (Stage V), test (Stage E; only after the val report's recorded
sha256) or the smoke's val stand-ins.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import so101_wm as sw  # noqa: E402
from embodied_jepa import so101_wm_data as sd  # noqa: E402
from embodied_jepa import so101_wm_runtime as runtime  # noqa: E402

ACTION_MANIFEST = ROOT / "configs" / "so101_sim_action.json"
SETS = ("true", "wrong", "zero", "hold")


# ----- shared ------------------------------------------------------------------------------------
def features_for(args, encoder: str) -> Path:
    path = args.features_pretrained if encoder == "pretrained" else args.features_random
    if path is None:
        raise SystemExit(f"no feature folder for the {encoder} encoder")
    return Path(path)


def open_eval_view(args, folder, moments):
    allow = False
    if args.eval_view == "test":
        if not (args.val_report and args.val_report_sha256):
            raise rt.GuardError("G-test: the test split opens only after the val report")
        if sd.sha256_file(args.val_report) != args.val_report_sha256:
            raise rt.GuardError("G-test: the val report differs from its recorded sha256")
        allow = True
    return sd.open_view(folder, args.eval_view, moments, allow_test=allow)


def job_id(report: dict) -> str:
    suffix = "-ext" if report.get("resumed_from") else ""
    return f"{report['arm']}-{report['seed']}{suffix}"


def read_jobs(paths) -> list[tuple[Path, dict]]:
    out = []
    for p in paths:
        report = json.loads((Path(p) / "report.json").read_text())
        if report.get("outcome") != "T-JOB-DONE":
            raise rt.GuardError(f"G-checkpoint: {p} did not finish")
        out.append((Path(p), report))
    ids = [job_id(r) for _, r in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("two jobs share an arm and seed")
    return out


def check_jobs(args, jobs) -> None:
    """§12: Stage V takes exactly the 15 (or extended) models; E takes the val report's."""
    if args.purpose not in ("val", "test"):
        return
    kinds = {r["kind"] for _, r in jobs}
    pairs = sorted((r["arm"], r["seed"]) for _, r in jobs)
    want = sorted((a, s) for a in sw.ARMS for s in sw.MODEL_SEEDS)
    if kinds != {"job"} or pairs != want:
        raise rt.GuardError("G-models: the gated stages take exactly the 15 models of §5")
    if args.purpose == "test":
        vr = json.loads(Path(args.val_report).read_text())
        kept = {k: v["checkpoint_sha256"] for k, v in vr["models"].items()}
        mine = {job_id(r): r["checkpoint_sha256"] for _, r in jobs}
        if kept != mine:
            raise rt.GuardError("G-models: Stage E takes the val report's checkpoints only")


def gpu_setup(report, device: str):
    import torch

    from embodied_jepa import devices

    report["device"] = device
    report["torch"] = torch.__version__
    if device == "cuda":
        rt.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")
    elif report.get("purpose") in ("val", "test"):
        raise SystemExit("the gated passes run on the GPU")
    return devices


def memmap_out(path, shape):
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def root_commands(view, starts, arm):
    commands = view.store.actions[starts[:, None] + np.arange(sw.ROOT_SPAN)]
    return runtime.command_set(commands, arm)


# ----- fitpred (GPU) -----------------------------------------------------------------------------
def cmd_fitpred(args) -> int:
    out = Path(args.output)
    report = runtime.header("fitpred", fit_view=args.fit_view, purpose=args.purpose)
    strict = args.purpose in ("val", "test")
    runtime.require_clean(report, strict)
    devices = gpu_setup(report, args.device)
    jobs = read_jobs(args.models)
    out.mkdir(parents=True, exist_ok=False)
    report["models"] = {}
    for job, jr in jobs:
        t0 = time.monotonic()
        name = job_id(jr)
        folder = features_for(args, jr["metadata"]["encoder"])
        moments, fit = runtime.load_fit(folder)
        if fit["report_sha256"] != jr["fit_report_sha256"]:
            raise rt.GuardError(f"G-features: {name} was trained on another feature fit")
        runtime.verify_features(folder, [sd.VIEWS[args.fit_view][0]])
        view = sd.open_view(folder, args.fit_view, moments)
        model, _ = runtime.load_job(job, args.device)
        starts, _ = view.roots()
        commands = root_commands(view, starts, jr["arm"])
        mdir = out / name
        mdir.mkdir()
        maps = {
            h: memmap_out(mdir / f"pred_fit_h{h}.npy", (len(starts), sw.LATENT_DIM))
            for h in sw.HORIZONS
        }
        for lo, pred in runtime.rollout_chunks(
            model, view.store.standardised(starts), commands, sw.HORIZONS
        ):
            for h in sw.HORIZONS:
                maps[h][lo : lo + len(pred[h])] = pred[h]
        files = {}
        for h, m in maps.items():
            m.flush()
            del m
            files[f"pred_fit_h{h}"] = sd.sha256_file(mdir / f"pred_fit_h{h}.npy")
        report["models"][name] = {
            "job": str(job),
            "checkpoint_sha256": jr["checkpoint_sha256"],
            "implementation_sha256": jr["implementation_sha256"],
            "roots": int(len(starts)),
            "files_sha256": files,
            "seconds": time.monotonic() - t0,
        }
        runtime.log(f"fitpred {name}: {len(starts)} roots, {time.monotonic() - t0:.0f} s")
        del model
    if args.device == "cuda":
        report["memory"] = devices.memory_report("cuda")
    runtime.require_clean(report, strict)
    report["finished_utc"] = runtime.utc()
    report["outcome"] = "FITPRED-DONE"
    runtime.write_json(out / "report.json", report)
    return 0


# ----- readouts (CPU) ----------------------------------------------------------------------------
def labels_at(view, index):
    s = view.store
    return sw.label_matrix(s.cube[index], s.jaw[index])


def cmd_readouts(args) -> int:
    out = Path(args.output)
    report = runtime.header("readouts", fit_view=args.fit_view, purpose=args.purpose)
    strict = args.purpose in ("val", "test")
    runtime.require_clean(report, strict)
    out.mkdir(parents=True, exist_ok=False)
    report["encoders"], report["models"] = {}, {}
    encoders = [
        e
        for e in sw.ENCODERS
        if (args.features_pretrained if e == "pretrained" else args.features_random)
    ]
    for encoder in encoders:
        t0 = time.monotonic()
        folder = features_for(args, encoder)
        moments, fit = runtime.load_fit(folder)
        runtime.verify_features(folder, [sd.VIEWS[args.fit_view][0]])
        view = sd.open_view(folder, args.fit_view, moments)
        rec = {"fit_report_sha256": fit["report_sha256"]}
        idx, ep = view.frame_index(sw.RENC_FRAME_STRIDE)
        x = view.store.standardised(idx)
        fits = sw.fit_readouts(x, labels_at(view, idx), view.episodes[ep])
        sw.save_readouts(out / f"renc_{encoder}.npz", fits)
        rec["renc"] = {k: sw.readout_record(v) for k, v in fits.items()}
        del x
        starts, ep = view.roots()
        x = view.store.standardised(starts)
        rec["copy"] = {}
        for h in sw.HORIZONS:
            fits = sw.fit_readouts(x, labels_at(view, starts + h), view.episodes[ep])
            sw.save_readouts(out / f"copy_{encoder}_h{h}.npz", fits)
            rec["copy"][str(h)] = {k: sw.readout_record(v) for k, v in fits.items()}
        rec["seconds"] = time.monotonic() - t0
        report["encoders"][encoder] = rec
        runtime.log(f"readouts {encoder}: {rec['seconds']:.0f} s")
    if args.fitpred:
        fp = Path(args.fitpred)
        fp_report = json.loads((fp / "report.json").read_text())
        if fp_report["fit_view"] != args.fit_view:
            raise rt.GuardError("G-readouts: fitpred ran on another fit view")
        report["fitpred_report_sha256"] = sd.sha256_file(fp / "report.json")
        for name, m in fp_report["models"].items():
            t0 = time.monotonic()
            jr = json.loads((Path(m["job"]) / "report.json").read_text())
            folder = features_for(args, jr["metadata"]["encoder"])
            moments, _ = runtime.load_fit(folder)
            view = sd.open_view(folder, args.fit_view, moments)
            starts, ep = view.roots()
            rec = {}
            for h in sw.HORIZONS:
                path = fp / name / f"pred_fit_h{h}.npy"
                if sd.sha256_file(path) != m["files_sha256"][f"pred_fit_h{h}"]:
                    raise rt.GuardError(f"G-readouts: {path} differs from fitpred's report")
                x = np.load(path, mmap_mode="r")
                fits = sw.fit_readouts(x, labels_at(view, starts + h), view.episodes[ep])
                sw.save_readouts(out / f"rpred_{name}_h{h}.npz", fits)
                rec[str(h)] = {k: sw.readout_record(v) for k, v in fits.items()}
            report["models"][name] = {"rpred": rec, "seconds": time.monotonic() - t0}
            runtime.log(f"readouts {name}: {time.monotonic() - t0:.0f} s")
    report["files_sha256"] = {p.name: sd.sha256_file(p) for p in sorted(out.glob("*.npz"))}
    runtime.require_clean(report, strict)
    report["finished_utc"] = runtime.utc()
    report["outcome"] = "READOUTS-DONE"
    runtime.write_json(out / "report.json", report)
    return 0


def load_readout_set(readouts: Path, files: dict, name: str) -> dict:
    path = readouts / name
    if sd.sha256_file(path) != files[name]:
        raise rt.GuardError(f"G-readouts: {name} differs from the readouts report")
    return sw.load_readouts(path)


# ----- evalpred (GPU) ----------------------------------------------------------------------------
def estimates(fits: dict, x) -> dict:
    return {label: sw.apply_readout(fits[label], x).astype(np.float32) for label in sw.LABELS}


def cmd_evalpred(args) -> int:
    out = Path(args.output)
    report = runtime.header("evalpred", eval_view=args.eval_view, purpose=args.purpose)
    strict = args.purpose in ("val", "test")
    runtime.require_clean(report, strict)
    devices = gpu_setup(report, args.device)
    jobs = read_jobs(args.models)
    check_jobs(args, jobs)
    readouts = Path(args.readouts)
    rr = json.loads((readouts / "report.json").read_text())
    report["readouts_report_sha256"] = sd.sha256_file(readouts / "report.json")
    out.mkdir(parents=True, exist_ok=False)
    report["models"] = {}
    tables = {}
    for job, jr in jobs:
        t0 = time.monotonic()
        name = job_id(jr)
        encoder = jr["metadata"]["encoder"]
        folder = features_for(args, encoder)
        moments, fit = runtime.load_fit(folder)
        if fit["report_sha256"] != jr["fit_report_sha256"]:
            raise rt.GuardError(f"G-features: {name} was trained on another feature fit")
        runtime.verify_features(folder, [sd.VIEWS[args.eval_view][0]])
        view = open_eval_view(args, folder, moments)
        starts, ep = view.roots()
        if "starts" not in tables:
            commands = view.store.actions[starts[:, None] + np.arange(sw.ROOT_SPAN)]
            cands = sw.candidate_sequences(commands, ep)
            tables = {
                "starts": starts,
                "wrong": sw.wrong_table(ep),
                "candidates": cands["sequences"],
                "candidate_classes": cands["classes"],
            }
            report["tables"] = {
                "roots": int(len(starts)),
                "episodes": int(len(view.episodes)),
                "starts_sha256": sw.table_sha256(starts),
                "wrong_sha256": sw.table_sha256(tables["wrong"]),
                "candidates_sha256": sw.table_sha256(cands["sequences"]),
            }
        elif not np.array_equal(tables["starts"], starts):
            raise rt.GuardError("G-roots: the encoders' views disagree on the roots")
        model, _ = runtime.load_job(job, args.device)
        rpred = {
            h: load_readout_set(readouts, rr["files_sha256"], f"rpred_{name}_h{h}.npz")
            for h in sw.HORIZONS
        }
        renc = load_readout_set(readouts, rr["files_sha256"], f"renc_{encoder}.npz")
        cmd = {
            "true": tables["candidates"][:, 0],
            "wrong": tables["candidates"][:, 0][tables["wrong"]],
            "zero": np.zeros_like(tables["candidates"][:, 0]),
            "hold": sw.hold_commands(tables["candidates"][:, 0]),
        }
        start_all = view.store.standardised(starts)
        result = {}
        mdir = out / name
        mdir.mkdir()
        pmaps = {
            h: memmap_out(mdir / f"pred_true_h{h}.npy", (len(starts), sw.LATENT_DIM))
            for h in sw.HORIZONS
        }
        for kind in SETS:
            commands = runtime.command_set(cmd[kind], jr["arm"])
            errs = {h: np.empty(len(starts)) for h in sw.HORIZONS}
            est = {(lab, h): [] for lab in sw.LABELS for h in sw.HORIZONS}
            est_enc = {(lab, h): [] for lab in sw.LABELS for h in sw.HORIZONS}
            for lo, pred in runtime.rollout_chunks(model, start_all, commands, sw.HORIZONS):
                n = len(pred[sw.HORIZONS[0]])
                for h in sw.HORIZONS:
                    target = view.store.standardised(starts[lo : lo + n] + h)
                    errs[h][lo : lo + n] = sw.latent_error(pred[h], target)
                    if kind == "true":
                        pmaps[h][lo : lo + n] = pred[h]
                        for lab, v in estimates(rpred[h], pred[h]).items():
                            est[(lab, h)].append(v)
                        for lab, v in estimates(renc, pred[h]).items():
                            est_enc[(lab, h)].append(v)
            for h in sw.HORIZONS:
                result[f"err_{kind}_h{h}"] = errs[h]
                if kind == "true":
                    for lab in sw.LABELS:
                        result[f"rpred_{lab}_h{h}"] = np.concatenate(est[(lab, h)])
                        result[f"renc_on_pred_{lab}_h{h}"] = np.concatenate(est_enc[(lab, h)])
        for m in pmaps.values():
            m.flush()
        del pmaps
        # §9: every candidate scored at h in {4, 8, 16}, by latent error and by R-pred's estimate
        hp = max(sw.PLAN_HORIZONS)
        for h in sw.PLAN_HORIZONS:
            result[f"cand_latent_h{h}"] = np.empty((len(starts), sw.CANDIDATES))
            for lab in sw.LABELS:
                width = 3 if lab == "cube" else 1
                result[f"cand_rpred_{lab}_h{h}"] = np.empty(
                    (len(starts), sw.CANDIDATES, width), np.float32
                )
        for j in range(sw.CANDIDATES):
            commands = runtime.command_set(tables["candidates"][:, j, :hp], jr["arm"])
            for lo, pred in runtime.rollout_chunks(model, start_all, commands, sw.PLAN_HORIZONS):
                n = len(pred[sw.PLAN_HORIZONS[0]])
                for h in sw.PLAN_HORIZONS:
                    target = view.store.standardised(starts[lo : lo + n] + h)
                    result[f"cand_latent_h{h}"][lo : lo + n, j] = sw.latent_error(pred[h], target)
                    for lab, v in estimates(rpred[h], pred[h]).items():
                        result[f"cand_rpred_{lab}_h{h}"][lo : lo + n, j] = v.reshape(n, -1)
        np.savez(mdir / "results.npz", **result)
        files = {p.name: sd.sha256_file(p) for p in sorted(mdir.iterdir())}
        report["models"][name] = {
            "job": str(job),
            "arm": jr["arm"],
            "seed": jr["seed"],
            "encoder": encoder,
            "checkpoint_sha256": jr["checkpoint_sha256"],
            "implementation_sha256": jr["implementation_sha256"],
            "files_sha256": files,
            "seconds": time.monotonic() - t0,
        }
        runtime.log(f"evalpred {name}: {len(starts)} roots, {time.monotonic() - t0:.0f} s")
        del model
    np.savez(
        out / "tables.npz",
        starts=tables["starts"],
        wrong=tables["wrong"],
        candidates=tables["candidates"],
    )
    report["tables"]["candidate_classes"] = tables["candidate_classes"]
    report["tables_file_sha256"] = sd.sha256_file(out / "tables.npz")
    if args.device == "cuda":
        report["memory"] = devices.memory_report("cuda")
    runtime.require_clean(report, strict)
    report["finished_utc"] = runtime.utc()
    report["outcome"] = "EVALPRED-DONE"
    runtime.write_json(out / "report.json", report)
    return 0


# ----- score (CPU) -------------------------------------------------------------------------------
class EvalData:
    """The evaluation view's encoded latents and labels at the roots, per encoder."""

    def __init__(self, args, encoder: str):
        folder = features_for(args, encoder)
        moments, fit = runtime.load_fit(folder)
        self.fit = fit
        self.view = open_eval_view(args, folder, moments)
        self.starts, self.ep = self.view.roots()
        self.basis = None
        if encoder == "pretrained":
            with np.load(folder / "basis.npz") as d:
                self.basis = d["basis"]
            if sw.basis_sha256(self.basis) != fit["basis_sha256"]:
                raise rt.GuardError("G-features: G1 (iii)'s basis differs from its record")
        s = self.view.store
        self.cube = {h: s.cube[self.starts + h].astype(np.float64) for h in (0, *sw.HORIZONS)}
        self.jaw = {h: s.jaw[self.starts + h].astype(np.float64) for h in (0, *sw.HORIZONS)}
        self.phase = {h: s.phase[self.starts + h] for h in sw.HORIZONS}
        self.visible = {h: s.visible[self.starts + h] for h in sw.HORIZONS}
        self.target_cache: dict = {}
        self.encoded_moments: dict = {}

    def truth(self, label, h):
        return self.cube[h] if label == "cube" else self.jaw[h]

    def targets(self, h):
        if h not in self.target_cache:
            self.target_cache = {h: self.view.store.standardised(self.starts + h)}
        return self.target_cache[h]


def readout_errors(fits, x, data: EvalData, h):
    return {
        lab: sw.label_errors(lab, sw.apply_readout(fits[lab], x), data.truth(lab, h))
        for lab in sw.LABELS
    }


def moving_masks(data: EvalData, h, thresholds):
    return {
        lab: sw.moving_mask(lab, data.truth(lab, 0), data.truth(lab, h), thresholds[lab])
        for lab in sw.LABELS
    }


def model_quantities(name, res, data: EvalData, boot, level, pred_path):
    """Per-horizon statistics of one model that do not need its twin."""
    out = {}
    for h in sw.HORIZONS:
        true = res[f"err_true_h{h}"]
        q = {
            "acc": float(true.sum()),
            "copy_ratio": sw.ratio_of_sums(true, res["copy_err"][h], boot, level),
            "copy_ratio_report": sw.ratio_of_sums(true, res["copy_err"][h], boot, sw.REPORT_LEVEL),
            "sens": sw.ratio_of_sums(res[f"err_wrong_h{h}"], true, boot, level),
            "zero_over_true": sw.ratio_of_sums(res[f"err_zero_h{h}"], true, boot, sw.REPORT_LEVEL),
            "hold_over_true": sw.ratio_of_sums(res[f"err_hold_h{h}"], true, boot, sw.REPORT_LEVEL),
            "mean_true_error": float(true.mean()),
        }
        pred = np.load(pred_path(h), mmap_mode="r")
        q["g1_stats"] = sw.collapse_statistics(pred, data.targets(h))
        errs = {}
        for lab in sw.LABELS:
            e = sw.label_errors(lab, res[f"rpred_{lab}_h{h}"], data.truth(lab, h))
            shift = sw.label_errors(lab, res[f"renc_on_pred_{lab}_h{h}"], data.truth(lab, h))
            errs[lab] = e
            q[f"rpred_{lab}"] = sw.median_stat(e, boot, level)
            q[f"renc_on_pred_{lab}"] = sw.median_stat(shift, boot, sw.REPORT_LEVEL)
            q[f"rpred_{lab}_by_phase"] = {
                str(int(p)): float(np.median(e[data.phase[h] == p]))
                for p in np.unique(data.phase[h])
            }
            vis = data.visible[h]
            q[f"rpred_{lab}_by_visibility"] = {
                cam: {
                    "visible": float(np.median(e[vis[:, i]])) if vis[:, i].any() else None,
                    "hidden": float(np.median(e[~vis[:, i]])) if (~vis[:, i]).any() else None,
                }
                for i, cam in enumerate(sw.CAMERAS)
            }
        cube = errs["cube"]
        q["rpred_cube_share_below_cm"] = {
            str(r): float((cube < r).mean()) for r in sw.ABSOLUTE_REFERENCES_CM
        }
        q["_errors"] = errs
        out[str(h)] = q
    # §9: planning usefulness
    plan = {}
    for h in sw.PLAN_HORIZONS:
        scores = res[f"cand_latent_h{h}"]
        rank = sw.true_rank(scores)
        cube = np.linalg.norm(res[f"cand_rpred_cube_h{h}"] - data.truth("cube", h)[:, None], axis=2)
        jaw = np.abs(res[f"cand_rpred_jaw_h{h}"][..., 0] - data.truth("jaw", h)[:, None])
        phys = cube + sw.PHYSICAL_COST_JAW_CM_PER_RAD * jaw
        prank = sw.true_rank(phys)
        entry = {
            "latent_top1": sw.mean_stat(rank == 0, boot, sw.REPORT_LEVEL),
            "latent_normalised_rank": sw.mean_stat(
                rank / (sw.CANDIDATES - 1), boot, sw.REPORT_LEVEL
            ),
            "physical_top1": sw.mean_stat(prank == 0, boot, sw.REPORT_LEVEL),
            "physical_normalised_rank": sw.mean_stat(
                prank / (sw.CANDIDATES - 1), boot, sw.REPORT_LEVEL
            ),
            "by_class": {},
        }
        classes = res["candidate_classes"]
        for cls in sorted(set(classes[1:])):
            cols = [j for j, c in enumerate(classes) if c == cls]
            sub = np.concatenate([scores[:, :1], scores[:, cols]], axis=1)
            psub = np.concatenate([phys[:, :1], phys[:, cols]], axis=1)
            entry["by_class"][cls] = {
                "latent_beats_all": float((sw.true_rank(sub) == 0).mean()),
                "physical_beats_all": float((sw.true_rank(psub) == 0).mean()),
            }
        plan[str(h)] = entry
    out["planning"] = plan
    return out


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def cmd_score(args) -> int:
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    report = runtime.header("score", purpose=args.purpose, eval_view=args.eval_view)
    strict = args.purpose in ("val", "test")
    runtime.require_clean(report, strict)
    level = sw.GATE_LEVEL
    if args.purpose == "grid":
        return score_grid(args, report, out)
    ep_dir = Path(args.evalpred)
    er = json.loads((ep_dir / "report.json").read_text())
    report["evalpred_report_sha256"] = sd.sha256_file(ep_dir / "report.json")
    if sd.sha256_file(ep_dir / "tables.npz") != er["tables_file_sha256"]:
        raise rt.GuardError("G-tables: tables.npz differs from its record")
    readouts = Path(args.readouts)
    rr = json.loads((readouts / "report.json").read_text())
    report["readouts_report_sha256"] = sd.sha256_file(readouts / "report.json")
    with np.load(ep_dir / "tables.npz") as t:
        candidates = t["candidates"]
    encoders = sorted({m["encoder"] for m in er["models"].values()})
    data = {e: EvalData(args, e) for e in encoders}
    first = data[encoders[0]]
    boot = sw.Bootstrap(first.view.episodes[first.ep])
    report["roots"] = int(len(first.starts))
    report["episodes"] = int(len(first.view.episodes))
    report["bootstrap"] = {
        "resamples": boot.resamples,
        "salt": sw.SALTS["bootstrap"],
        "gate_level": level,
    }
    if args.purpose == "calibrate":
        thresholds = dict(sw.MOVING_THRESHOLDS)
    else:
        thresholds = (
            sw.frozen_choices()["moving"]
            if args.purpose in ("val", "test")
            else (sw.MOVING or dict(sw.MOVING_THRESHOLDS))
        )
    report["moving_thresholds"] = thresholds
    # copy-last and the encoded-frame readout per encoder
    copy, renc_ceiling = {}, {}
    for e, d in data.items():
        start = d.view.store.standardised(d.starts)
        copy[e] = {"latent": {}, "readout": {}}
        for h in sw.HORIZONS:
            copy[e]["latent"][h] = sw.latent_error(start, d.targets(h))
            fits = load_readout_set(readouts, rr["files_sha256"], f"copy_{e}_h{h}.npz")
            copy[e]["readout"][h] = readout_errors(fits, start, d, h)
        renc = load_readout_set(readouts, rr["files_sha256"], f"renc_{e}.npz")
        renc_ceiling[e] = {}
        for h in sw.HORIZONS:
            errs = readout_errors(renc, d.targets(h), d, h)
            renc_ceiling[e][str(h)] = {
                lab: sw.median_stat(v, boot, sw.REPORT_LEVEL) for lab, v in errs.items()
            }
        idx, _ = d.view.frame_index()
        x = d.view.store.standardised(idx)
        allf = {
            lab: sw.label_errors(
                lab,
                sw.apply_readout(renc[lab], x),
                d.view.store.cube[idx] if lab == "cube" else d.view.store.jaw[idx],
            )
            for lab in sw.LABELS
        }
        renc_ceiling[e]["all_frames_median"] = {lab: float(np.median(v)) for lab, v in allf.items()}
        renc_ceiling[e]["all_frames"] = int(len(idx))
        del x
    report["renc_ceiling"] = renc_ceiling
    report["copy_readout"] = {
        e: {
            str(h): {lab: float(np.median(v)) for lab, v in c["readout"][h].items()}
            for h in sw.HORIZONS
        }
        for e, c in copy.items()
    }
    # per model
    quantities, results = {}, {}
    for name, m in er["models"].items():
        mdir = ep_dir / name
        for f, want in m["files_sha256"].items():
            if sd.sha256_file(mdir / f) != want:
                raise rt.GuardError(f"G-evalpred: {name}/{f} differs from its record")
        with np.load(mdir / "results.npz") as z:
            res = {k: z[k] for k in z.files}
        res["copy_err"] = copy[m["encoder"]]["latent"]
        res["candidate_classes"] = er["tables"]["candidate_classes"]
        results[name] = res
        quantities[name] = model_quantities(
            name, res, data[m["encoder"]], boot, level, lambda h, d=mdir: d / f"pred_true_h{h}.npy"
        )
        quantities[name]["meta"] = {
            k: m[k] for k in ("arm", "seed", "encoder", "checkpoint_sha256")
        }
        runtime.log(f"score {name}")
    # ranking ceiling (§9)
    with open(ACTION_MANIFEST) as fh:
        manifest = json.load(fh)
    d0 = data["pretrained"] if "pretrained" in data else first
    start_target = d0.view.store.gripper_target[d0.starts]
    ceiling = {}
    for h in sw.PLAN_HORIZONS:
        jt = sw.jaw_targets(
            candidates[:, :, :h, sw.GRIPPER_CHANNEL],
            np.repeat(start_target[:, None], sw.CANDIDATES, 1),
            closed=manifest["gripper_closed_rad"],
            opened=manifest["gripper_open_rad"],
            rate=manifest["gripper_max_delta_per_step_rad"],
        )[..., -1]
        same_jaw = (np.abs(jt - jt[:, :1]) <= sw.TIE_TOLERANCE).all(1)
        ceiling[str(h)] = {
            "one_minus_sequence_tie_share": 1.0 - sw.tie_share(candidates, h),
            "share_all_candidates_same_jaw_target": float(same_jaw.mean()),
        }
    report["ranking_ceiling"] = ceiling
    # moving subsets
    moving = {}
    for h in sw.HORIZONS:
        masks = moving_masks(first, h, thresholds)
        moving[str(h)] = {
            lab: {"roots": int(mk.sum()), "share": float(mk.mean())} for lab, mk in masks.items()
        }
    report["moving_subsets"] = moving
    # pairwise: eligible X against N_X of the same seed (G1 iii, G3, G5 b/c)
    pairs = {}
    for name, q in quantities.items():
        arm, seed = q["meta"]["arm"], q["meta"]["seed"]
        if arm not in sw.NO_ACTION_TWIN:
            continue
        twin = next(
            (
                n
                for n, t in quantities.items()
                if t["meta"]["arm"] == sw.NO_ACTION_TWIN[arm]
                and t["meta"]["seed"] == sw.twin_seed(seed)
            ),
            None,
        )
        if twin is None:
            continue
        d = data["pretrained"]
        entry = {"x": name, "n_x": twin}
        for h in sw.HORIZONS:
            px = np.load(ep_dir / name / f"pred_true_h{h}.npy", mmap_mode="r")
            pn = np.load(ep_dir / twin / f"pred_true_h{h}.npy", mmap_mode="r")
            ne = len(d.view.episodes)
            mx = sw.ProjectedMoments(px, d.ep, d.basis, ne)
            mn = sw.ProjectedMoments(pn, d.ep, d.basis, ne)
            if h not in d.encoded_moments:
                d.encoded_moments[h] = sw.ProjectedMoments(d.targets(h), d.ep, d.basis, ne)
            g1iii = sw.projected_rank_difference(mx, mn, d.encoded_moments[h], boot, level)
            full_n = quantities[twin][str(h)]["g1_stats"]["effective_rank_ratio"]
            g1iii["full_width_difference"] = (
                quantities[name][str(h)]["g1_stats"]["effective_rank_ratio"] - full_n
            )
            masks = moving_masks(d, h, thresholds)
            g5 = {}
            ex, en = quantities[name][str(h)]["_errors"], quantities[twin][str(h)]["_errors"]
            for lab in sw.LABELS:
                g5[lab] = {
                    "median": quantities[name][str(h)][f"rpred_{lab}"],
                    "vs_copy": sw.ratio_of_medians(
                        ex[lab], copy["pretrained"]["readout"][h][lab], boot, level, masks[lab]
                    ),
                    "vs_noaction": sw.ratio_of_medians(ex[lab], en[lab], boot, level, masks[lab]),
                }
            entry[str(h)] = {
                "g1_stats": quantities[name][str(h)]["g1_stats"],
                "g1_iii": g1iii,
                "copy_ratio": quantities[name][str(h)]["copy_ratio"],
                "noaction_ratio": sw.ratio_of_sums(
                    results[name][f"err_true_h{h}"], results[twin][f"err_true_h{h}"], boot, level
                ),
                "sens": quantities[name][str(h)]["sens"],
                "g2_allocation_0p8": quantities[name][str(h)]["copy_ratio"]["ci"][1]
                <= sw.G["G2_reported_allocation"],
                "g5": g5,
            }
        pairs[name] = entry
        runtime.log(f"pair {name} / {twin}")
    # §9: the encoder floor reading (L against R, seed by seed)
    floor = {}
    for name, q in quantities.items():
        if q["meta"]["arm"] != "L":
            continue
        rname = next(
            (
                n
                for n, t in quantities.items()
                if t["meta"]["arm"] == "R" and t["meta"]["seed"] == q["meta"]["seed"]
            ),
            None,
        )
        if rname is None:
            continue
        floor[name] = {
            str(h): sw.paired_median_difference(
                q[str(h)]["_errors"]["cube"],
                quantities[rname][str(h)]["_errors"]["cube"],
                boot,
                sw.REPORT_LEVEL,
            )
            for h in sw.GATE_HORIZONS
        }
    report["encoder_floor_l_minus_r_cube"] = floor
    report["models"] = strip(quantities)
    report["pairs"] = strip(pairs)
    if args.purpose == "calibrate":
        report["calibration"] = calibrate(
            pairs, quantities, renc_ceiling, moving, data["pretrained"], ep_dir, boot
        )
    elif args.purpose in ("val", "test", "smoke"):
        report |= gate_and_row(args, pairs, moving, quantities, er)
    report["models_checkpoints"] = {
        n: q["meta"]["checkpoint_sha256"] for n, q in quantities.items()
    }
    runtime.require_clean(report, strict)
    report["finished_utc"] = runtime.utc()
    report["outcome"] = {"val": "VAL-DONE", "test": "E-DONE"}.get(
        args.purpose, f"{args.purpose.upper()}-DONE"
    )
    if args.purpose == "val":
        report["models"] = {
            n: dict(q, checkpoint_sha256=q["meta"]["checkpoint_sha256"])
            for n, q in report["models"].items()
        }
    sha = runtime.write_json(out, report)
    runtime.log(f"{report['outcome']}: {out} sha256 {sha}")
    print(sha)
    return 0


def calibrate(pairs, quantities, renc_ceiling, moving, d, ep_dir, boot) -> dict:
    """§10.4-§10.5 and §7.4's subset rule on D0-eval: the bars from calibration L (against
    calibration N_L for the truncation check), with every other value disclosed."""
    name = next(n for n, q in quantities.items() if q["meta"]["arm"] == "L")
    entry = pairs[name]
    rank = {h: entry[str(h)]["g1_stats"]["effective_rank_ratio"] for h in sw.GATE_HORIZONS}
    std = {h: entry[str(h)]["g1_stats"]["std_ratio"] for h in sw.GATE_HORIZONS}
    bars = sw.g1_bars(rank, std)
    bars["B_rank_before_truncation_check"] = bars["B_rank"]
    trunc = {}
    raised = False
    twin = entry["n_x"]
    for h in sw.GATE_HORIZONS:
        px = np.load(ep_dir / name / f"pred_true_h{h}.npy")
        pn = np.load(ep_dir / twin / f"pred_true_h{h}.npy", mmap_mode="r")
        mn = sw.ProjectedMoments(pn, d.ep, d.basis, len(d.view.episodes))
        for k in sw.G["G1_truncations"]:
            t = sw.truncate(px, k)
            stats = sw.collapse_statistics(t, d.targets(h))
            mt = sw.ProjectedMoments(t, d.ep, d.basis, len(d.view.episodes))
            iii = sw.projected_rank_difference(mt, mn, d.encoded_moments[h], boot, sw.GATE_LEVEL)
            passes = sw.g1_passes(stats, iii["ci"][0], bars)
            trunc[f"h{h}_k{k}"] = {"stats": stats, "g1_iii": iii, "passes": passes}
    passing = [v["stats"]["effective_rank_ratio"] for v in trunc.values() if v["passes"]["all"]]
    if passing:
        bars["B_rank"] = sw.ceil_above_01(max(passing))
        raised = True
    bars["truncation_raised_B_rank"] = raised
    b5 = sw.g5_bars(renc_ceiling["pretrained"]["all_frames_median"])
    shares = {lab: moving["8"][lab]["share"] for lab in sw.LABELS}
    g4 = {str(h): entry[str(h)]["sens"] for h in sw.GATE_HORIZONS}
    g4_flag = any(entry[str(h)]["sens"]["ci"][0] < sw.G["G4_min_sens"] for h in sw.GATE_HORIZONS)
    bars_all = {"B_rank": bars["B_rank"], "B_std": bars["B_std"], **b5}
    g5_miss = {
        f"h{h}_{lab}": bool(entry[str(h)]["g5"][lab]["median"]["ci"][1] > bars_all[f"B_{lab}"])
        for h in sw.GATE_HORIZONS
        for lab in sw.LABELS
    }
    disclosed = {str(h): sw.gates_for(entry[str(h)], bars_all) for h in sw.GATE_HORIZONS}
    return {
        "calibration_model": name,
        "calibration_noaction_model": twin,
        "g1_bars": bars,
        "truncation_check": trunc,
        "g5_bars": b5,
        "bars": bars_all,
        "moving_thresholds": sw.moving_thresholds_from_d0(shares),
        "g4_calibration_sens": g4,
        "g4_lb_below_1p10": g4_flag,
        "g5_calibration_misses_bar": g5_miss,
        "calibration_gate_flags_disclosed": disclosed,
    }


def gate_and_row(args, pairs, moving, quantities, er) -> dict:
    """The gates per eligible arm, seed and h, G6 per seed, and (test) the row of §11."""
    if args.purpose == "smoke":
        bars = sw.BARS or {"B_rank": 0.1, "B_std": 0.25, "B_cube": 1.0, "B_jaw": 0.1}
    else:
        bars = sw.frozen_choices()["bars"]
    gates, g6, small = {}, {}, []
    for name, entry in pairs.items():
        arm, seed = quantities[name]["meta"]["arm"], str(quantities[name]["meta"]["seed"])
        gates.setdefault(arm, {})[seed] = {
            str(h): sw.gates_for(entry[str(h)], bars) for h in sw.GATE_HORIZONS
        }
        curves = {}
        for n in (name, entry["n_x"]):
            jr = json.loads((Path(er["models"][n]["job"]) / "report.json").read_text())
            curves[n] = jr["training"]["val_curve"]
        sat = {n: sw.saturation(c) for n, c in curves.items()}
        g6.setdefault(arm, {})[seed] = all(s["passes"] for s in sat.values())
        gates[arm][seed]["g6_detail"] = sat
    for h in sw.GATE_HORIZONS:
        for lab in sw.LABELS:
            if moving[str(h)][lab]["roots"] < sw.MOVING_MIN_TEST_ROOTS:
                small.append(f"M_{lab},{h}: {moving[str(h)][lab]['roots']} roots")
    out = {"bars_used": bars, "gates": gates, "g6": g6}
    if args.purpose in ("test", "smoke"):
        complete = all(
            arm in gates and len(gates[arm]) == len(sw.MODEL_SEEDS) for arm in sw.ELIGIBLE
        )
        if complete:
            out["row"] = sw.decide(void_reasons=[], small_subsets=small, gates=gates, g6=g6)
        else:
            out["row"] = {
                "row": "NOT-COMPUTED",
                "reason": "the smoke has fewer seeds than §5",
                "small_subsets": small,
            }
    return out


def score_grid(args, report, out) -> int:
    """§10.3: R-enc (fitted on D0-fit, every fifth frame) on every D0-eval frame, g4 and g8."""
    readouts = {g: Path(p) for g, p in (("g4", args.readouts), ("g8", args.readouts_g8))}
    folders = {"g4": Path(args.features_pretrained), "g8": Path(args.features_g8)}
    medians = {}
    for g in ("g4", "g8"):
        rr = json.loads((readouts[g] / "report.json").read_text())
        renc = load_readout_set(readouts[g], rr["files_sha256"], "renc_pretrained.npz")
        moments, fit = runtime.load_fit(folders[g])
        view = sd.open_view(folders[g], args.eval_view, moments)
        idx, _ = view.frame_index()
        x = view.store.standardised(idx)
        errs = {
            lab: sw.label_errors(
                lab,
                sw.apply_readout(renc[lab], x),
                view.store.cube[idx] if lab == "cube" else view.store.jaw[idx],
            )
            for lab in sw.LABELS
        }
        medians[g] = {lab: float(np.median(v)) for lab, v in errs.items()} | {
            "frames": int(len(idx))
        }
        medians[g]["penalties"] = {
            lab: rr["encoders"]["pretrained"]["renc"][lab]["penalty"] for lab in sw.LABELS
        }
        del x
    report["renc_d0_eval"] = medians
    report["rule"] = sw.grid_rule(medians["g4"]["cube"], medians["g8"]["cube"])
    report["finished_utc"] = runtime.utc()
    report["outcome"] = "GRID-DONE"
    sha = runtime.write_json(out, report)
    runtime.log(f"grid: {report['rule']} ({sha})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "pass_", metavar="pass", choices=("fitpred", "readouts", "evalpred", "score")
    )
    parser.add_argument(
        "--purpose", required=True, choices=("calibrate", "grid", "smoke", "val", "test")
    )
    parser.add_argument("--features-pretrained", default=None)
    parser.add_argument("--features-random", default=None)
    parser.add_argument("--features-g8", default=None, help="grid: the g8 pretrained folder")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--fit-view", default="fit-sample")
    parser.add_argument("--eval-view", default="val")
    parser.add_argument("--fitpred", default=None)
    parser.add_argument("--readouts", default=None)
    parser.add_argument("--readouts-g8", default=None, help="grid: the g8 readouts")
    parser.add_argument("--evalpred", default=None)
    parser.add_argument("--val-report", default=None)
    parser.add_argument("--val-report-sha256", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", help="cpu only for development dry-runs")
    args = parser.parse_args(argv)
    if args.purpose in ("val", "test"):
        if args.pass_ in ("fitpred", "readouts") and args.fit_view != "fit-sample":
            raise SystemExit(f"{args.purpose}: the fit view is the fit sample")
        if args.pass_ in ("evalpred", "score") and args.eval_view != args.purpose:
            raise SystemExit(f"{args.purpose}: the evaluation view is {args.purpose}")
    if args.purpose in ("calibrate", "grid") and "test" in (args.fit_view, args.eval_view):
        raise SystemExit("Stage 0 never opens test")
    return {
        "fitpred": cmd_fitpred,
        "readouts": cmd_readouts,
        "evalpred": cmd_evalpred,
        "score": cmd_score,
    }[args.pass_](args)


if __name__ == "__main__":
    sys.exit(main())
