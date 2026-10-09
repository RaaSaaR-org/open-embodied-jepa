"""TASK-087 Stages V and E: evaluate the kept models on val (the val report and δ) or on test (the
gated evaluation and the row). Protocol §6.3-§9.

``--split val`` writes the val report (outcome VAL-DONE) with every model's statistics and δ_h.
``--split test`` needs the val report and its sha256 and Stage F''s test report; it writes the
test report with the comparisons, the criteria, the row and the reported-only diagnostics.
``--debug`` (smoke only) accepts debug-seed models and lets val stand in for test.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grounded_wm as gw  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_models(paths, features_report_sha256, *, debug, device):
    import torch

    from embodied_jepa import grounded_wm_model as gm

    models = {}
    for path in paths:
        path = Path(path)
        rep = json.loads((path / "report.json").read_text())
        if rep.get("outcome") != "T-JOB-DONE":
            raise rt.GuardError(f"G-model: {path} is not T-JOB-DONE")
        if rep.get("kind") != "job" and not debug:
            raise rt.GuardError(f"G-model: {path} is a {rep.get('kind')} run")
        if rep["features_report_sha256"] != features_report_sha256:
            raise rt.GuardError(f"G-model: {path} was trained on other features")
        if gd.sha256_file(path / "model.pt") != rep["checkpoint_sha256"]:
            raise rt.GuardError(f"G-model: {path}/model.pt differs from its sha256")
        arm, seed = rep["arm"], int(rep["seed"])
        gw.check_model_seed(seed, debug=debug)
        ckpt = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = gm.GroundedWM(arm, config=ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        if (arm, seed) in models:
            raise rt.GuardError(f"G-model: two models for {arm}-{seed}")
        models[(arm, seed)] = {
            "model": model,
            "report": str(path / "report.json"),
            "report_sha256": gd.sha256_file(path / "report.json"),
            "checkpoint_sha256": rep["checkpoint_sha256"],
            "kept_update": rep["training"]["kept_update"],
            "last_two_triggered": rep["training"]["last_two_triggered"],
            "parameters": rep["parameters"],
        }
    return models


def ridge_fit(x, y, lam):
    xtx = x.T @ x
    xty = x.T @ y
    return np.linalg.solve(xtx + lam * np.eye(len(xtx)), xty)


def r2(pred, y):
    return float(1.0 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def pair_features(store, starts):
    a = np.asarray(store.latents[starts], np.float32).reshape(len(starts), -1)
    b = np.asarray(store.latents[starts + 1], np.float32).reshape(len(starts), -1)
    return np.concatenate([a, b], 1).astype(np.float64)


def inverse_probe(train, val, val_starts, test, test_starts, fit_seeds) -> dict:
    """§9: ridge from encoded (z_t, z_{t+1}) to the 7 played dims; train fit sample (every 5th
    transition), penalty chosen on val, R² on test."""
    dims = list(gw.PLAYED_ACTION_DIMS)
    starts = []
    for row in train.table:
        if row["seed"] in fit_seeds:
            starts.append(row["offset"] + np.arange(0, row["frames"] - 1, 5))
    starts = np.concatenate(starts)
    x = pair_features(train, starts)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x = (x - mu) / sd
    y = train.actions[starts][:, dims].astype(np.float64)
    xv = (pair_features(val, val_starts) - mu) / sd
    yv = val.actions[val_starts][:, dims]
    grid = [10.0**k for k in range(0, 7)]
    scores = {}
    for lam in grid:
        w = ridge_fit(x, y - y.mean(0), lam)
        scores[lam] = r2(xv @ w + y.mean(0), yv)
    best = max(scores, key=scores.get)
    w = ridge_fit(x, y - y.mean(0), best)
    xt = (pair_features(test, test_starts) - mu) / sd
    yt = test.actions[test_starts][:, dims]
    return {
        "fit_pairs": int(len(starts)),
        "lambda": best,
        "val_r2": {str(k): v for k, v in scores.items()},
        "test_r2": r2(xt @ w + y.mean(0), yt),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="the report file")
    parser.add_argument("--val-report", default=None)
    parser.add_argument("--val-report-sha256", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    import torch

    from embodied_jepa import devices
    from embodied_jepa import grounded_wm_model as gm

    report = {
        "task": gw.TASK,
        "protocol": gw.PROTOCOL,
        "split": args.split,
        "debug": bool(args.debug),
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    rt.assert_local_import(ROOT, report)
    if args.device == "cuda":
        rt.gpu_guard(report, min_free_gib=4.0, require_lock=not args.debug)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
    feat = Path(args.features)
    feat_sha = gd.sha256_file(feat / "report.json")
    report["features_report_sha256"] = feat_sha
    eval_split = args.split
    if args.split == "test":
        if args.val_report is None or args.val_report_sha256 is None:
            raise rt.GuardError("G-test: the test evaluation needs the val report and its sha256")
        if gd.sha256_file(args.val_report) != args.val_report_sha256:
            raise rt.GuardError("G-test: the val report differs from its recorded sha256")
        val_report = json.loads(Path(args.val_report).read_text())
        if val_report.get("outcome") != "VAL-DONE" or (val_report["debug"] and not args.debug):
            raise rt.GuardError("G-test: not a real VAL-DONE report")
        if val_report["features_report_sha256"] != feat_sha:
            raise rt.GuardError("G-test: the val report used other features")
        report["val_report_sha256"] = args.val_report_sha256
        delta = val_report["delta"]
        if (feat / "report_test.json").exists():
            ft = json.loads((feat / "report_test.json").read_text())
            if ft.get("outcome") != "FT-DONE" or ft["val_report_sha256"] != args.val_report_sha256:
                raise rt.GuardError("G-test: the test features were not made after this val report")
            report["test_features_report_sha256"] = gd.sha256_file(feat / "report_test.json")
        elif args.debug:
            eval_split = "val"  # smoke: val stands in for test
        else:
            raise rt.GuardError("G-test: no test features")
    report["evaluated_split"] = eval_split
    store = gd.FeatureStore(feat, eval_split)
    starts, episode = store.roots()
    wrong = gw.wrong_table(episode)
    table = gw.candidate_table(episode)
    start, commands, states, targets = store.root_arrays(starts)
    report["roots"] = {
        "n": int(len(starts)),
        "episodes": int(len(np.unique(episode))),
        "wrong_sha256": gw.table_sha256(wrong),
        "candidates_sha256": gw.table_sha256(table),
    }
    weights = gw.cluster_bootstrap(episode)
    models = load_models(args.models, feat_sha, debug=args.debug, device=args.device)
    report["models"] = {
        f"{a}-{s}": {k: v for k, v in m.items() if k != "model"} for (a, s), m in models.items()
    }
    seeds = sorted({s for _, s in models})
    t0 = time.monotonic()
    errors, preds8 = {}, {}
    for (arm, seed), entry in sorted(models.items()):
        model = entry["model"]
        flags = gw.check_arm(arm)
        st = states if (flags["state_token"] or flags["fusion"]) else None

        def predict(s0, cmd, _st, hs, model=model, st=st):
            return gm.predict_at(model, s0, cmd, st, hs, chunk=2048, device=args.device)

        errors[(arm, seed)], true = gw.per_root_errors(
            predict, start, commands, targets, st, wrong, table
        )
        preds8[(arm, seed)] = true[8]
        log(f"{arm}-{seed} evaluated ({time.monotonic() - t0:.0f} s)")
    level = gw.REPORT_LEVEL
    report["statistics"] = {
        f"{a}-{s}": gw.summarise(e, weights, level) for (a, s), e in errors.items()
    }
    if args.split == "val":
        if all(("P", s) in errors for s in gw.MODEL_SEEDS) and not args.debug:
            val_true = {
                s: {h: errors[("P", s)][h]["true"] for h in gw.GATE_HORIZONS}
                for s in gw.MODEL_SEEDS
            }
        else:  # smoke: the debug seeds stand in
            ps = [s for a, s in errors if a == "P"]
            mapping = dict(zip(gw.MODEL_SEEDS, (ps * 3)[:3], strict=True))
            val_true = {
                s: {h: errors[("P", mapping[s])][h]["true"] for h in gw.GATE_HORIZONS}
                for s in gw.MODEL_SEEDS
            }
        report["delta"] = gw.delta_from_val(val_true, weights)
        report["outcome"] = "VAL-DONE"
    else:
        comparisons, criteria, separation = {}, {}, {}
        for (arm, seed), e in errors.items():
            if arm in ("P", "N") or ("P", seed) not in errors:
                continue
            p = errors[("P", seed)]
            comparisons[f"{arm}-{seed}"] = gw.compare(e, p, weights)
            separation[f"{arm}-{seed}"] = {
                str(h): gw.mean_difference(
                    e[h]["wrong"] - e[h]["true"], p[h]["wrong"] - p[h]["true"], weights, level
                )
                for h in gw.HORIZONS
            }
            criteria.setdefault(arm, {})[str(seed)] = {
                str(h): gw.criteria_met(comparisons[f"{arm}-{seed}"], delta, h)
                for h in gw.GATE_HORIZONS
            }
        plain_checks = {}
        for seed in seeds:
            if ("P", seed) in errors and ("N", seed) in errors:
                p, n = errors[("P", seed)], errors[("N", seed)]
                p_vs_n = {"8": gw.ratio_stat(p[8]["true"], n[8]["true"], weights, level)}
                plain_checks[str(seed)] = gw.plain_valid(
                    report["statistics"][f"P-{seed}"], p_vs_n
                ) | {"p_over_n_h8": p_vs_n["8"]}
        report["comparisons"] = comparisons
        report["criteria"] = criteria
        report["plain_checks"] = plain_checks
        report["separation_vs_P"] = separation
        report["delta"] = delta
        void = []
        expected = {(a, s) for a in gw.ARMS for s in gw.MODEL_SEEDS}
        if not args.debug and set(models) != expected:
            void.append("not every arm and seed was evaluated")
        if args.debug:
            # smoke: map the debug seeds onto the protocol's seed names for the row logic
            mapping = dict(zip(gw.MODEL_SEEDS, (seeds * 3)[:3], strict=True))
            crit = {
                a: {
                    str(s): criteria.get(a, {}).get(
                        str(mapping[s]), {str(h): {"all": False} for h in gw.GATE_HORIZONS}
                    )
                    for s in gw.MODEL_SEEDS
                }
                for a in gw.ELIGIBLE
            }
            checks = {
                str(s): plain_checks.get(str(mapping[s]), {"valid": False, "reasons": ["missing"]})
                for s in gw.MODEL_SEEDS
            }
        else:
            crit = {a: criteria.get(a, {}) for a in gw.ELIGIBLE}
            checks = plain_checks
        row = gw.decide(void_reasons=void, plain_checks=checks, criteria=crit)
        d_top1 = {
            a: {
                str(s): comparisons.get(f"{a}-{s}", {})
                .get("1", {})
                .get("d_top1", {})
                .get("value", float("nan"))
                for s in gw.MODEL_SEEDS
            }
            for a in gw.ELIGIBLE
        }
        report["row"] = row
        report["phase3_model"] = gw.phase3_model(row, d_top1)
        # ----- reported only ------------------------------------------------------------------
        extra = {}
        extra["effective_rank_h8"] = {
            "encoded": gw.effective_rank(targets[8]),
            **{f"{a}-{s}": gw.effective_rank(p) for (a, s), p in preds8.items()},
        }
        local = starts - store.offsets[episode]
        extra["tie_share"] = {
            str(h): {
                "all": gw.tie_share(commands, table, h),
                "start_0": gw.tie_share(commands, table[local == 0], h),
                "later": gw.tie_share(commands, table[local > 0], h),
            }
            for h in gw.HORIZONS
        }
        train = gd.FeatureStore(feat, "train")
        val = gd.FeatureStore(feat, "val")
        vs, _ = val.roots()
        fit_seeds = {r["seed"] for r in train.table if r["seed"] % gw.FIT_SAMPLE_MODULUS == 0}
        extra["inverse_probe"] = inverse_probe(train, val, vs, store, starts, fit_seeds)
        heads = {}
        dims = list(gw.PLAYED_ACTION_DIMS)
        y = commands[:, 0][:, dims]
        for (arm, seed), entry in models.items():
            model = entry["model"]
            flags = gw.check_arm(arm)
            with torch.no_grad():
                z0 = torch.as_tensor(start, device=args.device)
                z1 = torch.as_tensor(targets[1], device=args.device)
                h1 = torch.as_tensor(
                    preds_h1 := gm.predict_at(
                        model,
                        start,
                        commands,
                        states if (flags["state_token"] or flags["fusion"]) else None,
                        (1,),
                        chunk=2048,
                        device=args.device,
                    )[1],
                    device=args.device,
                )
                rec = {}
                if flags["inverse"]:
                    rec["inverse_r2_encoded_pairs"] = r2(model.inverse(z0, z1).cpu().numpy(), y)
                    rec["inverse_r2_predicted_pairs"] = r2(model.inverse(z0, h1).cpu().numpy(), y)
                if flags["state_head"]:
                    sp = gm.predict_states(
                        model, start, commands, states, gw.HORIZONS, device=args.device
                    )
                    sn = model.norm_state(torch.as_tensor(states, device=args.device)).cpu().numpy()
                    for h in gw.HORIZONS:
                        later = (
                            model.norm_state(
                                torch.as_tensor(store.state[starts + h], device=args.device)
                            )
                            .cpu()
                            .numpy()
                        )
                        rec[f"state_mse_h{h}"] = float(((sp[h] - later) ** 2).mean())
                        rec[f"state_mse_nochange_h{h}"] = float(((sn - later) ** 2).mean())
                if flags["readout"]:
                    pm = model.palm_mean.cpu().numpy()
                    ps = model.palm_scale.cpu().numpy()
                    for h in (1, 8):
                        truth = (store.palm[starts + h] - pm) / ps
                        p8 = torch.as_tensor(
                            preds8[(arm, seed)] if h == 8 else preds_h1, device=args.device
                        )
                        enc = torch.as_tensor(targets[h], device=args.device)
                        rec[f"palm_mse_predicted_h{h}"] = float(
                            ((model.readout_head(p8.flatten(1)).cpu().numpy() - truth) ** 2).mean()
                        )
                        rec[f"palm_mse_encoded_h{h}"] = float(
                            ((model.readout_head(enc.flatten(1)).cpu().numpy() - truth) ** 2).mean()
                        )
                if rec:
                    heads[f"{arm}-{seed}"] = rec
        extra["heads"] = heads
        report["reported_only"] = extra
        report["outcome"] = "E-DONE"
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = utc()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    log(f"{report['outcome']}: {report.get('row', {}).get('row', '')} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
