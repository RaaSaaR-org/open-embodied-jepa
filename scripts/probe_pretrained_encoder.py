"""TASK-063 gated run: a frozen pretrained encoder against its random-init floor
(``apple_pretrained_encoder_v1``).

Order (protocol §10 guards first, then §9 anchors, then §5-§11):

1. **Guards** -- G-hash (every pinned file, clean tree, CPU only), G-split (TASK-061's 190 roots and
   fold hash), TASK-061's G-render, G-expert, G-look and G-prior (run by TASK-061's own ``measure``
   and checks), G-frames (the post-look frames equal TASK-062's). Any failure, crash or global
   wall-cap stop writes a report with outcome ``V`` and a ``void_reason``.
2. **Anchors (G-anchor)** -- L-raw, L-E0, L-random through TASK-061's code against TASK-061
   run-1's report; F-tok and E0-tok through TASK-062's stage code against TASK-062 run-1's report.
3. **G-weights, G-repro** -- the pinned DINOv2 files and both weight digests; a second forward
   pass of the pretrained encoder is bit-identical.
4. **Features and readouts** (CPU) -- P-cls, P-tok, R-cls, R-tok (decisional arms and floors),
   P-mean, R-mean (reported only), each through TASK-061's ``evaluate_source``. An arm whose
   encoders produce a non-finite feature, or whose time exceeds the per-arm cap, is "not
   evaluated"; that never voids the run.
5. **Decision** -- floors, p-values (``encoder_study.arm_pvalues``), Holm over two arms, the
   apple-hidden check, the first matching row (§11); diagnostics and paired comparisons.

    uv run --no-sync python scripts/probe_pretrained_encoder.py \
        --output outputs/task063-pretrained-encoder/run-1

``--smoke`` checks the wiring on 24 roots (20 train, 4 val) with the readout TARGETS REPLACED by
seeded noise, so it reveals nothing about the real readouts. It skips G-prior, G-anchor, G-frames,
the fold hash and the clean-tree check (full-cohort or gated-run facts).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import encoder_study as es  # noqa: E402
from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa import observation_reprobe as orp  # noqa: E402
from embodied_jepa import pretrained_encoder as pe  # noqa: E402
from embodied_jepa import pretrained_study as ps  # noqa: E402
from embodied_jepa.contracts import ContractError  # noqa: E402

PROTOCOL = "apple_pretrained_encoder_v1"
TASK = "TASK-063"
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-pretrained-encoder-v1.json"
TASK059_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1.json"
TASK061_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-observation-reprobe-v1.json"
TASK061_REPORT = ROOT / "outputs" / "task061-observation-reprobe" / "run-1" / "report.json"
TASK062_REPORT = ROOT / "outputs" / "task062-encoder-study" / "run-1" / "report.json"
STORE = ROOT / "data" / "apple-wide-v1"
WEIGHTS = pe.DEFAULT_DIRECTORY


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded unchanged; G-hash pins their bytes.
BASE = _load("_probe_info_ceiling", "scripts/probe_info_ceiling.py")
REPROBE = _load("_probe_observation_reprobe", "scripts/probe_observation_reprobe.py")
CAL = _load("_calibrate_encoder_study", "scripts/calibrate_encoder_study.py")
STUDY = _load("_probe_encoder_study", "scripts/probe_encoder_study.py")
VoidRun = BASE.VoidRun


def sha256(path) -> str:
    return BASE.sha256(path)


def write_report(path: Path, report: dict) -> None:
    BASE.write_report(path, report)


def _stats(values) -> dict:
    import torch

    from embodied_jepa.models.base import _statistics

    return _statistics(torch.from_numpy(np.asarray(values, np.float64)))


# The rows of the protocol's §3.3 label-free table, recomputed in the run (§8).
LABEL_FREE_STAGES = ("patch_embedding_tokens", "block6_tokens", "tokens", "cls", "tokens_mean")


def token_mean(tokens) -> np.ndarray:
    """Mean of the 256 final patch tokens (P-mean / R-mean), float64."""
    return np.asarray(tokens).reshape(len(tokens), pe.GRID * pe.GRID, pe.WIDTH).mean(1)


def featurise(model, frames_by_key: dict) -> tuple[dict | None, float, str | None]:
    """Read-outs of every frame set; ``None`` and a reason on a non-finite feature."""
    begin = time.monotonic()
    try:
        out = {key: pe.features(model, frames) for key, frames in frames_by_key.items()}
    except ContractError as error:
        if "non-finite" in str(error):
            return None, time.monotonic() - begin, str(error)
        raise
    for value in out.values():
        value["mean"] = token_mean(value["tokens"])
    return out, time.monotonic() - begin, None


def recomputed_view(result: dict, preds: dict) -> dict:
    return STUDY.recomputed_view(result, preds)


def reference_views(report: dict, sources) -> dict:
    return {s: es.anchor_view(report["results"], report["per_root"], s) for s in sources}


def run(output: Path, *, smoke: bool = False, device: str = ps.DEVICE) -> dict:
    import torch
    import transformers

    from embodied_jepa.training import peak_rss_bytes, source_identity

    clock = BASE.Clock(ps.GLOBAL_WALL_SECONDS)
    report = {
        "protocol": PROTOCOL,
        "task": TASK,
        "smoke": smoke,
        "source": source_identity(),
        "environment": BASE.environment()
        | {
            "device": device,
            "transformers": transformers.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "status": "running",
        "runner_sha256": {
            "scripts/probe_pretrained_encoder.py": sha256(Path(__file__)),
            "src/embodied_jepa/pretrained_study.py": sha256(Path(ps.__file__)),
        },
        "learned_apple_to_plate_successes": 0,
        "exemption_spent": False,
    }
    manifest = json.loads(MANIFEST.read_text())
    try:
        # --- G-hash, clean tree, device ---
        actual = {
            name: sha256(ROOT / name) if (ROOT / name).is_file() else None
            for name in manifest["hashes"]
        }
        ic.check_hashes(manifest["hashes"], actual)
        report["hashes"] = actual
        if device != manifest["budget"]["device"]:
            raise ic.GuardError(f"device {device} != the manifest's {manifest['budget']['device']}")
        if not smoke:
            ic.check_clean_tree(report["source"]["dirty"])
        task059 = json.loads(TASK059_MANIFEST.read_text())
        task061 = json.loads(TASK061_MANIFEST.read_text())
        # --- G-split (TASK-061's) ---
        roots, _collector = BASE.plan_roots()
        dataset = json.loads((STORE / "meta" / "jepa_manifest.json").read_text())
        expected = ic.EXPECTED_ROOTS
        if smoke:
            train_r = [r for r in roots if r["split"] == "train"][:20]
            val_r = [r for r in roots if r["split"] == "val"][:4]
            roots = sorted(train_r + val_r, key=lambda r: r["seed"])
            expected = {"train": 20, "val": 4}
        ic.check_split(roots, dataset["splits"], expected=expected)
        seeds = [r["seed"] for r in roots]
        fold = ic.fold_of(len(roots))
        fold_hash = ic.fold_assignment_sha256(seeds, fold)
        if not smoke:
            ic.check_fold_hash(fold_hash, task061["split"]["fold_assignment_sha256"])
        report["fold_assignment_sha256"] = fold_hash
        # --- frames: TASK-061's measure, with G-render, G-expert, G-look ---
        reader = BASE.FrameReader(dataset, [r["episode_id"] for r in roots])
        rows, frames = REPROBE.measure(roots, reader, clock=clock)
        n = len(rows)
        ic.check_render(
            [r["path"]["P0_reset_112_equals_stored"] for r in rows], frames["reset_proprio"]
        )
        ic.check_expert(
            [r["expert_step0"] for r in rows],
            [r["recorded_step0"] for r in rows],
            [r["aim_offset"] for r in rows],
            [r["apple_xy_label"] for r in rows],
            [r["apple_xy_truth"] for r in rows],
        )
        report["look"] = orp.check_look(
            sequence_sha=orp.sequence_sha256(orp.look_sequence()),
            want_sha=task061["look_motion"]["sequence_sha256_float32_bytes"],
            applied=frames["applied"],
            post_states=frames["post_state"],
            apple_xy_moves=[r["apple_xy_move_m"] for r in rows],
            plate_moves=[r["plate_move_m"] for r in rows],
            contacts=[r["hand_contact"] for r in rows],
            steps_checked=[r["look_steps_checked"] for r in rows],
        )
        xy = np.array([r["apple_xy_truth"] for r in rows], np.float64)
        dx = np.array([r["expert_post_look"][0] for r in rows], np.float64)
        dy = np.array([r["expert_post_look"][1] for r in rows], np.float64)
        dx_reset = np.array([r["expert_step0"][0] for r in rows], np.float64)
        train_mask = np.array([r["split"] == "train" for r in rows])
        occ059 = np.array([r["apple_pixels_reset"]["onboard_112"] == 0 for r in rows])
        # --- G-prior ---
        if not smoke:
            orp.check_priors(
                ic.prior_summary(xy, dx_reset, occ059, fold, train_mask),
                task059["calibration"]["values"]["prior_baselines"],
                "reset targets vs TASK-059",
            )
            orp.check_priors(
                ic.prior_summary(xy, dx, occ059, fold, train_mask),
                task061["calibration"]["values"]["prior_baselines"]["post_look_targets"],
                "post-look targets",
            )
        ablation_ok = int(sum(r["ablation_reproduces"]["L"] for r in rows))
        report["ablation_renderer_reproduces_L"] = ablation_ok
        post_sha = hashlib.sha256(frames["post_112"].tobytes()).hexdigest()
        report["post_look_frames_sha256"] = post_sha
        if not smoke:  # --- G-frames ---
            ps.check_frames(post_sha, manifest["calibration"]["post_look_frames_sha256"])
        if smoke:
            noise = np.random.default_rng(0)
            xy = xy.mean(0) + noise.normal(0, 0.017, xy.shape)
            dx = np.clip(noise.normal(0.1, 0.4, n), -0.4, 0.4)
            dx[dx == 0] = 0.1
            dy = noise.normal(-0.39, 0.02, n)
            report["smoke_targets"] = "seeded noise; no real target reached a readout"
        priors = ic.prior_predictions(xy, dx, occ059, fold, dy=dy)
        split_priors = ic.prior_predictions(xy, dx, occ059, np.where(train_mask, 1, 0), dy=dy)
        masks = {
            k: m
            for k, m in {
                "all": np.ones(n, bool),
                "reset_occluded": occ059,
                "reset_visible": ~occ059,
            }.items()
            if m.any()
        }
        flag = np.array([r["apple_pixels_post_look"]["onboard_112"] == 0 for r in rows])
        if flag.any() and (~flag).any():
            masks |= {"arm_occluded": flag, "arm_visible": ~flag}
        results, kept, preds, readout_seconds, feature_sha = {}, {}, {}, {}, {}

        def single(name, x):
            clock.check(f"readout {name}")
            begin = time.monotonic()
            feature_sha[name] = hashlib.sha256(
                np.ascontiguousarray(np.asarray(x, np.float64)).tobytes()
            ).hexdigest()
            g, diag = ic.gram(x)
            results[name], kept[name], preds[name] = BASE.evaluate_source(
                name, g, diag, xy, dx, dy, fold, priors, masks, train_mask, split_priors
            )
            results[name]["feature_dim"] = int(np.asarray(x).reshape(len(x), -1).shape[1])
            readout_seconds[name] = time.monotonic() - begin

        # --- stage 1: anchors (G-anchor) ---
        clock.check("anchors")
        features, _digests059 = BASE.encoder_features(
            {"post_112": frames["post_112"], "post_112_hidden": frames["post_112_hidden"]},
            task059,
        )
        flat = lambda a: a.reshape(len(a), -1) / 255.0  # noqa: E731
        single("L_raw", flat(frames["post_112"]))
        single("L_E0", features[("E0", "post_112")])
        single("L_random", features[("random", "post_112")])
        config, store, _settings = STUDY.open_config()
        refs, digests = STUDY.reference_models(config, store, task061)
        report["reference_encoder_weights_sha256"] = digests
        ref_tokens = {}
        for name, model in refs.items():
            stages = STUDY.stage_features(model, frames["post_112"])
            label = "L_E0" if name == "E0" else "L_random"
            same = np.array_equal(stages["probe"], features[(name, "post_112")])
            report.setdefault("stage_code_equals_task061_features", {})[label] = bool(same)
            if not same:
                raise ic.GuardError(f"G-anchor: stage features of {name} differ from TASK-061's")
            ref_tokens[name] = stages["tokens"]
        del refs
        single("E0_tok", ref_tokens["E0"])
        single("F_tok", ref_tokens["random"])
        del ref_tokens
        if not smoke:
            anchors = {}
            for label, path in (("task061", TASK061_REPORT), ("task062", TASK062_REPORT)):
                sources = ps.ANCHOR_SETS[label]
                anchors[label] = ps.anchor_compare(
                    reference_views(json.loads(path.read_text()), sources),
                    {s: recomputed_view(results[s], preds[s]) for s in sources},
                    sources,
                )
            report["anchor"] = anchors
            for label, comparison in anchors.items():
                ps.check_anchor(comparison, label)
        else:
            report["anchor"] = {"verdict": "skipped (smoke)"}
        # --- G-weights, G-repro ---
        clock.check("weights")
        pretrained = pe.load_pretrained(WEIGHTS)
        floor = pe.random_init(WEIGHTS)
        weights = {
            "files_sha256": pe.check_files(WEIGHTS),
            "pretrained_digest": pe.weights_digest(pretrained),
            "floor_digest": pe.weights_digest(floor),
        }
        report["weights"] = weights
        ps.check_weights(weights["pretrained_digest"], weights["floor_digest"], manifest)
        keys = {"post_112": frames["post_112"], "post_112_hidden": frames["post_112_hidden"]}
        feats, feat_seconds, failures = {}, {}, {}
        for name, model in (("P", pretrained), ("R", floor)):
            clock.check(f"features {name}")
            feats[name], feat_seconds[name], failures[name] = featurise(model, keys)
        if feats["P"] is not None:
            again = pe.features(pretrained, frames["post_112"])
            ps.check_repro({k: feats["P"]["post_112"][k] for k in again}, again)
            report["G_repro"] = "bit-identical"
        report["feature_seconds"] = feat_seconds
        report["feature_failures"] = failures
        # --- stage 2: readouts ---
        for enc in ("P", "R"):
            if feats[enc] is None:
                continue
            for point, label in (("cls", "cls"), ("tokens", "tok"), ("mean", "mean")):
                single(f"{enc}_{label}", feats[enc]["post_112"][point])
        # --- stage 3: arms ---
        arms, pvalues = {}, {}
        for arm in ps.ARMS:
            name, floor_name = ps.SOURCE[arm], ps.SOURCE[ps.ARM_FLOOR[arm]]
            finite = feats["P"] is not None and feats["R"] is not None
            seconds = feat_seconds["P"] + feat_seconds["R"]
            seconds += readout_seconds.get(name, 0.0) + readout_seconds.get(floor_name, 0.0)
            evaluated = finite and not ps.over_cap(seconds)
            entry = {
                "source": name,
                "floor": floor_name,
                "readout_point": ps.ARM_POINT[arm],
                "seconds": seconds,
                "per_arm_cap_seconds": ps.PER_ARM_SECONDS,
                "non_finite_features": not finite,
                "state": ps.arm_state(evaluated=evaluated),
            }
            if evaluated:
                entry["succeeds"] = bool(results[name]["all"]["succeeds"])
                entry["beats_prior"] = bool(results[name]["all"]["beats_prior"])
                entry["floor_comparison"] = ic.beats_random_floor(
                    kept[name]["all"], kept[floor_name]["all"]
                )
                entry["beats_floor"] = bool(entry["floor_comparison"]["beats_random_floor"])
                entry["pvalues"] = es.arm_pvalues(
                    preds[name]["xy"],
                    preds[name]["dx"],
                    xy,
                    dx,
                    priors,
                    kept[floor_name]["all"],
                    kept[name]["all"],
                )
            else:
                entry |= {
                    "succeeds": False,
                    "beats_prior": False,
                    "beats_floor": False,
                    "pvalues": {"p": 1.0, "reason": "arm not evaluated"},
                }
            pvalues[arm] = entry["pvalues"]["p"]
            arms[arm] = entry
        holm = ps.holm(pvalues)
        reproduces = ablation_ok == n
        for arm in ps.ARMS:
            name, entry = ps.SOURCE[arm], arms[arm]
            point = ps.ARM_POINT[arm]
            hidden_eval = None
            if entry["state"] != "evaluated":
                entry["spurious_check"] = {
                    "available": False,
                    "spurious": True,
                    "reading": "arm not evaluated; no spurious check",
                    "hidden_all_roots": None,
                }
            elif reproduces:
                g_new, norms = ic.cross_gram(
                    feats["P"]["post_112_hidden"][point], feats["P"]["post_112"][point]
                )
                xy_hat = orp.predict_with_fold_readouts(preds[name]["xy_fits"], fold, g_new, norms)
                dx_hat = orp.predict_with_fold_readouts(preds[name]["dx_fits"], fold, g_new, norms)
                hidden_eval = ic.evaluate(xy_hat, dx_hat[:, 0], xy, dx, priors, np.ones(n, bool))
            if entry["state"] == "evaluated":
                entry["spurious_check"] = orp.spurious_verdict(
                    hidden_eval, renderer_reproduces=reproduces
                ) | {"hidden_all_roots": ic.public(hidden_eval) if hidden_eval else None}
            entry["spurious"] = bool(entry["spurious_check"]["spurious"])
            entry["holm_rejected"] = bool(holm["rejected"][arm])
            entry["passes"] = ps.passes(
                evaluated=entry["state"] == "evaluated",
                succeeds=entry["succeeds"],
                beats_floor=entry["beats_floor"],
                rejected=entry["holm_rejected"],
                spurious=entry["spurious"],
            )
            entry["qualifier"] = ps.qualifier(
                passes=entry["passes"],
                succeeds=entry["succeeds"],
                beats_floor=entry["beats_floor"],
                rejected=entry["holm_rejected"],
                beats_prior=entry["beats_prior"],
            )
            entry["p"] = entry["pvalues"]["p"]
        report["arms"] = arms
        report["holm"] = holm
        decision = ps.decide(void=False, arms=arms)
        decision["abandonment_clause_fires"] = ps.abandonment_fires(decision["outcome"])
        decision["also_matching_rows"] = ps.also_matching(decision["outcome"], arms)
        report["decision"] = decision
        # --- reported: diagnostics, label-free table, paired comparisons ---
        report["diagnostics"] = {
            f"{enc}_{point}": _stats(feats[enc]["post_112"][point])
            for enc in ("P", "R")
            if feats[enc] is not None
            for point in ("cls", "tokens", "mean")
        }
        cal_frames, _plates = CAL.render_post_look(roots)
        if not np.array_equal(cal_frames["post"], frames["post_112"]):
            raise ic.GuardError("G-render: the calibration renderer's post-look frames differ")
        label_free = {
            "raw_pixels": CAL.shares(
                flat(cal_frames["post"]).astype(np.float64),
                flat(cal_frames["apple_hidden"]).astype(np.float64),
                flat(cal_frames["plate_hidden"]).astype(np.float64),
            )
        }
        for enc, model in (("P", pretrained), ("R", floor)):
            if feats[enc] is None:
                continue
            staged = {
                k: pe.features(model, cal_frames[k], hidden_states=True)
                for k in ("post", "apple_hidden", "plate_hidden")
            }
            for stage in LABEL_FREE_STAGES:
                label_free[f"{enc}_{stage}"] = CAL.shares(
                    staged["post"][stage],
                    staged["apple_hidden"][stage],
                    staged["plate_hidden"][stage],
                )
        report["label_free"] = label_free
        pairs = [
            ("P_cls", "R_cls"),
            ("P_tok", "R_tok"),
            ("P_cls", "L_raw"),
            ("P_tok", "L_raw"),
            ("P_cls", "F_tok"),
            ("P_tok", "F_tok"),
            ("P_cls", "L_E0"),
            ("P_tok", "E0_tok"),
            ("R_tok", "F_tok"),
            ("P_mean", "P_cls"),
            ("P_mean", "R_mean"),
        ]
        report["pairwise_reported"] = {
            f"{a}_minus_{b}": {
                s: ic.compare_sources(kept[a][s], kept[b][s])
                for s in ("all", "reset_occluded", "reset_visible")
                if s in kept[a] and s in kept[b]
            }
            for a, b in pairs
            if a in kept and b in kept
        }
        report["readout_seconds"] = readout_seconds
        report["feature_sha256"] = feature_sha
        report["results"] = results
        report["per_root"] = [
            {
                "seed": r["seed"],
                "split": r["split"],
                "fold": int(fold[i]),
                "reset_occluded": bool(occ059[i]),
                "apple_xy": [float(v) for v in xy[i]],
                "expert_dx_post_look": float(dx[i]),
                "predictions": {
                    s: {
                        "xy": preds[s]["xy"][i].tolist(),
                        "dx": float(preds[s]["dx"][i]),
                        "dy": float(preds[s]["dy"][i]),
                    }
                    for s in preds
                },
            }
            for i, r in enumerate(rows)
        ]
        clock.check("report")
        report["status"] = "complete"
        report["outcome"] = decision["outcome"]
    except (ic.GuardError, pe.WeightsError, VoidRun) as error:
        report["status"] = "void"
        report["decision"] = {"outcome": "V", "reason": str(error)}
        report["outcome"] = "V"
        report["void_reason"] = f"guard: {error}"
    except Exception as error:  # noqa: BLE001 - a crash is recorded as V, then re-raised
        report["status"] = "void: crashed before the report was complete"
        report["decision"] = {"outcome": "V", "reason": f"{type(error).__name__}: {error}"}
        report["outcome"] = "V"
        report["void_reason"] = f"exception: {type(error).__name__}: {error}"
        report["traceback"] = "".join(traceback.format_exception(error))
        report["elapsed_seconds"] = clock.elapsed()
        report["peak_rss_bytes"] = peak_rss_bytes()
        write_report(output / "report.json", report)
        raise
    report["elapsed_seconds"] = clock.elapsed()
    report["peak_rss_bytes"] = peak_rss_bytes()
    write_report(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = run(args.output, smoke=args.smoke)
    print(json.dumps({k: report.get(k) for k in ("status", "decision", "elapsed_seconds")}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
