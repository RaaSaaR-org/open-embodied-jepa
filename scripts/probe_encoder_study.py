"""TASK-062 gated run: the cross-fitted encoder study (``apple_encoder_study_v1``).

Order (protocol §11 guards first, then §9 anchors, then training, then §5-§10):

1. **Guards** -- G-hash (every pinned file; episode files checked before decoding; encoder
   digests; clean tree), G-split (TASK-061's 190 roots and fold hash, plus the cross-fitting
   rules for every training episode), and TASK-061's G-render, G-expert, G-look and G-prior, run by
   TASK-061's own ``measure`` and checks. Any failure, crash or global wall-cap stop writes a
   report with outcome ``V`` and a ``void_reason``.
2. **Anchors (G-anchor), before any training** -- L-raw, L-E0 and L-random recomputed through
   TASK-061's code and compared with TASK-061 run-1's report. Then the cheap readout test of
   E-pool on E0 itself (E0-tok, reported only) and the token floor F-tok.
3. **Training** -- R0, SIG, REC and PLAIN, each as e(A) and e(B), on MPS, from the seed-0
   init, on the train-split episodes of the other half's train roots. An arm-level failure
   (per-encoder cap, non-finite loss) makes the arm "not evaluated"; it never voids the run.
4. **Features and diagnostics** on CPU -- probe feature and final tokens on the post-look and
   apple-hidden frames; collapse statistics; action sensitivity; the label-free table.
5. **Readouts** -- cross-fitted nested CV (``encoder_study.crossfit_nested_cv``), floors, p_F,
   Holm over A-tok, A-sig, A-rec, A-plain, the apple-hidden check, the decision (§10).

    uv run --no-sync python scripts/probe_encoder_study.py \
        --output outputs/task062-encoder-study/run-1

``--smoke`` checks the wiring on 24 roots (20 train, 4 val) with a few training steps and the
readout TARGETS REPLACED by seeded noise, so it reveals nothing about the real readouts. It skips
G-prior, G-anchor, the fold hash and the clean-tree check (full-cohort or gated-run facts).
"""

from __future__ import annotations

import argparse
import copy
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

PROTOCOL = "apple_encoder_study_v1"
TASK = "TASK-062"
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-encoder-study-v1.json"
TASK059_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1.json"
TASK061_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-observation-reprobe-v1.json"
TASK061_REPORT = ROOT / "outputs" / "task061-observation-reprobe" / "run-1" / "report.json"
STORE = ROOT / "data" / "apple-wide-v1"


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded unchanged; G-hash pins their bytes.
BASE = _load("_probe_info_ceiling", "scripts/probe_info_ceiling.py")
REPROBE = _load("_probe_observation_reprobe", "scripts/probe_observation_reprobe.py")
CAL = _load("_calibrate_encoder_study", "scripts/calibrate_encoder_study.py")
VoidRun = BASE.VoidRun


def sha256(path) -> str:
    return BASE.sha256(path)


def write_report(path: Path, report: dict) -> None:
    BASE.write_report(path, report)


class ArmFailure(Exception):
    """An arm-level failure (§11 'not guards'): demotes the arm, never voids the run."""


# ----- encoders --------------------------------------------------------------------------------
def open_config():
    from embodied_jepa.world_model_v2 import _open

    config, store, settings, cameras = _open(BASE.CONFIG)
    if tuple(cameras) != ("onboard_rgb",):
        raise ic.GuardError(f"config cameras {cameras} != ('onboard_rgb',)")
    return config, store, dict(settings)


def reference_models(config, store, task061):
    """Frozen E0 and the seed-0 random init, through TASK-061's loader, digests checked."""
    measure = _load("_measure", "scripts/measure_policy_offline_conditionals.py")
    models, digests = {}, {}
    for name, arm in (("E0", "a2"), ("random", "a1")):
        _policy, source = measure.build_policy(BASE.CHECKPOINTS / f"{arm}.pt", config, store, "cpu")
        digest = source.weights_sha256()
        ic.check_encoder_digest(name, digest, task061["encoders"][name]["encoder_weights_sha256"])
        source.model.eval()
        models[name], digests[name] = source.model, digest
    return models, digests


def stage_features(model, frames) -> dict:
    """Probe feature and final patch tokens (float64), through the calibration's stage code."""
    stages = CAL.encoder_stages(model, frames)
    return {"probe": stages["probe_feature"], "tokens": stages["final_tokens"]}


def provenance(store, source, name, half):
    from embodied_jepa.world_model_v2 import _provenance

    return _provenance(store, source, PROTOCOL) | {"encoder_study_model": name, "half": half}


def build_model(backend, store, settings, name, device, metadata):
    from embodied_jepa.config import MODELS

    return MODELS.create(
        backend,
        state_schema=store.state_schema,
        device=device,
        seed=es.MODEL_SEED,
        config=settings | es.MODEL_OVERRIDES[name],
        metadata=metadata,
    )


def load_half(store, episodes, workers=8, check_budget=lambda: None):
    """Decode exactly ``episodes`` (train split only) with world model v2's own decoder."""
    from embodied_jepa.world_model_v2 import load_split

    subset = copy.copy(store)
    subset.manifest = dict(store.manifest)
    subset.manifest["splits"] = dict(store.manifest["splits"]) | {"train": list(episodes)}
    arrays = load_split(
        subset,
        "train",
        ("onboard_rgb",),
        workers=workers,
        check_budget=check_budget,
        acknowledge_privileged_training_labels=True,
    )
    if list(arrays.episode_ids) != list(episodes):
        raise ic.GuardError("G-split: decoded episodes differ from the requested ones")
    return arrays


def train_encoder(
    name,
    half,
    arrays,
    store,
    settings,
    backend,
    directory,
    clock,
    *,
    steps,
    device,
    cap_seconds,
    source,
):
    """One encoder of the recipe (§5.1); returns its record. Raises ArmFailure on an arm-level
    failure and VoidRun on the global wall cap."""
    import torch

    from embodied_jepa.contracts import ContractError
    from embodied_jepa.world_model_v2 import batch, cosine_lr, window_starts

    tag = f"{name}_{half}"
    paths = {
        "checkpoint": directory / f"{tag}.pt",
        "decoder": directory / f"{tag}.decoder.pt",
        "curves": directory / f"{tag}.metrics.jsonl",
    }
    if any(p.exists() for p in paths.values()):
        raise FileExistsError(f"refusing to overwrite {tag}")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = provenance(store, source, name, half)
    model = build_model(backend, store, settings, name, device, metadata)
    init_digest = es.check_init(model, readout_heads=model.config["readout_heads"])  # G-init
    mean, std = es.welford_moments(arrays.states, arrays.mask)
    model.fit_state_normalization(mean, std, training_episode_ids=arrays.episode_ids)
    decoder = optimizer = None
    if name == "REC":
        decoder = es.make_decoder(model.config["latent_dim"]).to(device)
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=model.config["learning_rate"],
            weight_decay=model.config["weight_decay"],
        )
    windows = window_starts(arrays, es.HORIZON)
    sampler = np.random.default_rng(es.SAMPLER_SEED[half])
    base_lr = model.config["learning_rate"]
    begin = time.monotonic()
    record = {
        "model": name,
        "half": half,
        "device": device,
        "steps_requested": steps,
        "train_windows": int(len(windows)),
        "init_weights_digest": init_digest,
        "overrides": es.MODEL_OVERRIDES[name],
        "image_sigreg_weight": es.IMAGE_SIGREG_WEIGHT.get(name, 0.0),
        "reconstruction_weight": es.RECONSTRUCTION_WEIGHT.get(name, 0.0),
        "readout_targets": name != "PLAIN",
        "completed_steps": 0,
        "status": "running",
    }

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    try:
        for step in range(1, steps + 1):
            clock.check(f"training {tag} step {step}")
            if time.monotonic() - begin > cap_seconds:
                raise ArmFailure(f"per-encoder cap {cap_seconds} s reached at step {step}")
            lr = cosine_lr(base_lr, step, steps, es.FINAL_LR_FRACTION)
            for group in model.optimizer.param_groups:
                group["lr"] = lr
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr
            picked = windows[sampler.integers(len(windows), size=es.BATCH_SIZE)]
            sequence, targets = batch(arrays, picked, es.HORIZON, store.state_schema)
            try:
                if name in ("SIG", "REC"):
                    metrics = es.study_train_step(
                        model,
                        sequence,
                        targets,
                        image_sigreg_weight=es.IMAGE_SIGREG_WEIGHT.get(name, 0.0),
                        decoder=decoder,
                        decoder_optimizer=optimizer,
                        reconstruction_weight=es.RECONSTRUCTION_WEIGHT.get(name, 0.0),
                    )
                elif name == "PLAIN":
                    metrics = model.train_step(sequence, readout_targets=None)
                else:
                    metrics = model.train_step(sequence, readout_targets=targets)
            except ContractError as error:
                if "non-finite" in str(error):
                    raise ArmFailure(f"non-finite loss or gradient at step {step}") from error
                raise
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ArmFailure(f"non-finite training metrics at step {step}")
            record["completed_steps"] = step
            if step % 25 == 0 or step == 1:
                sync()
                with paths["curves"].open("a") as stream:
                    stream.write(
                        json.dumps(
                            {"step": step, "lr": lr, "elapsed_s": time.monotonic() - begin}
                            | metrics,
                            sort_keys=True,
                        )
                        + "\n"
                    )
        sync()
        record["status"] = "completed"
    except ArmFailure as failure:
        record["status"] = f"failed: {failure}"
    record["elapsed_seconds"] = time.monotonic() - begin
    model.metadata["runner_state"] = {"step": record["completed_steps"], "half": half}
    model.save(paths["checkpoint"])
    record["checkpoint"] = str(paths["checkpoint"])
    record["checkpoint_sha256"] = sha256(paths["checkpoint"])
    record["encoder_digest"] = es.weights_digest(model.model.encoder)
    record["curves"] = str(paths["curves"])
    if decoder is not None:
        torch.save(decoder.state_dict(), paths["decoder"])
        record["decoder_sha256"] = sha256(paths["decoder"])
    record["metadata"] = metadata
    return record


def reload_cpu(record, backend, store, settings):
    model = build_model(backend, store, settings, record["model"], "cpu", record["metadata"])
    model.load(Path(record["checkpoint"]))
    model.eval()
    if es.weights_digest(model.model.encoder) != record["encoder_digest"]:
        raise ic.GuardError(f"reloaded encoder {record['model']}_{record['half']} digest differs")
    return model


def collapse_statistics(model, arrays) -> dict:
    """``_statistics`` of the probe feature and flattened tokens on 1024 training-half frames."""
    import torch

    from embodied_jepa.models.base import _statistics

    rng = np.random.default_rng(es.COLLAPSE_SEED)
    rows = np.sort(rng.choice(len(arrays.states), size=es.COLLAPSE_FRAMES, replace=False))
    feats = stage_features(model, arrays.frames["onboard_rgb"][rows])
    return {k: _statistics(torch.from_numpy(v)) for k, v in feats.items()}


def action_sensitivity(model, arrays, store) -> dict:
    """Median over 512 windows of ||z8(a) - z8(a_perm)|| / ||z8(a) - z8_true|| (§8)."""
    import torch

    from embodied_jepa.world_model_v2 import batch, window_starts

    rng = np.random.default_rng(es.SENSITIVITY_SEED)
    windows = window_starts(arrays, es.SENSITIVITY_HORIZON)
    picked = windows[np.sort(rng.choice(len(windows), size=es.SENSITIVITY_WINDOWS, replace=False))]
    perm = es.derangement(len(picked), rng)
    ratios, shifts, errors = [], [], []
    model.eval()
    with torch.no_grad(), model.rng_scope():
        for start in range(0, len(picked), 64):
            chunk = picked[start : start + 64]
            sequence, _targets = batch(arrays, chunk, es.SENSITIVITY_HORIZON, store.state_schema)
            latents = model.observe_sequence(sequence)
            actions = model.sequence_tensors(sequence)
            other_seq, _ = batch(
                arrays, picked[perm[start : start + 64]], es.SENSITIVITY_HORIZON, store.state_schema
            )
            other = model.sequence_tensors(other_seq)
            true_roll = model.rollout(latents[:, 0], actions)[:, -1]
            perm_roll = model.rollout(latents[:, 0], other)[:, -1]
            shift = (true_roll - perm_roll).norm(dim=-1)
            error = (true_roll - latents[:, -1]).norm(dim=-1)
            shifts.append(shift.cpu().numpy())
            errors.append(error.cpu().numpy())
            ratios.append((shift / error.clamp_min(1e-12)).cpu().numpy())
    ratios, shifts, errors = (
        np.concatenate(x).astype(np.float64) for x in (ratios, shifts, errors)
    )
    return {
        "windows": int(len(ratios)),
        "horizon": es.SENSITIVITY_HORIZON,
        "median_ratio": float(np.median(ratios)),
        "median_shift": float(np.median(shifts)),
        "median_error": float(np.median(errors)),
    }


# ----- the run ---------------------------------------------------------------------------------
def anchor_reference(task061_report: dict) -> dict:
    return {
        s: es.anchor_view(task061_report["results"], task061_report["per_root"], s)
        for s in es.ANCHOR_SOURCES
    }


def recomputed_view(result: dict, preds: dict) -> dict:
    keep = {k: v for k, v in result.items() if k in es._ANCHOR_RESULT_KEYS}
    per_root = [
        {"xy": preds["xy"][i].tolist(), "dx": float(preds["dx"][i]), "dy": float(preds["dy"][i])}
        for i in range(len(preds["dx"]))
    ]
    clean, _found = BASE.finite_json({"results": keep, "per_root": per_root})
    return json.loads(json.dumps(clean, allow_nan=False))


def run(output: Path, *, smoke: bool = False, smoke_steps: int = 3, device: str = "mps") -> dict:
    from embodied_jepa.training import peak_rss_bytes, source_identity

    clock = BASE.Clock(es.GLOBAL_WALL_SECONDS)
    report = {
        "protocol": PROTOCOL,
        "task": TASK,
        "smoke": smoke,
        "source": source_identity(),
        "environment": BASE.environment() | {"training_device": device},
        "status": "running",
        "runner_sha256": {
            "scripts/probe_encoder_study.py": sha256(Path(__file__)),
            "src/embodied_jepa/encoder_study.py": sha256(Path(es.__file__)),
        },
    }
    manifest = json.loads(MANIFEST.read_text())
    try:
        # --- G-hash, clean tree ---
        actual = {
            name: sha256(ROOT / name) if (ROOT / name).is_file() else None
            for name in manifest["hashes"]
        }
        ic.check_hashes(manifest["hashes"], actual)
        report["hashes"] = actual
        if not smoke:
            ic.check_clean_tree(report["source"]["dirty"])
            if device != manifest["budget"]["training_device"]:
                raise ic.GuardError(f"training device {device} != the manifest's")
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
        # --- G-split (cross-fitting) ---
        episodes_by_half = {
            h: es.training_episodes(dataset, es.training_roots(roots, fold, h)) for h in es.HALVES
        }
        report["training_split"] = es.check_training_split(dataset, roots, fold, episodes_by_half)
        report["training_split"]["episode_ids"] = episodes_by_half
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
        report["post_look_frames_sha256"] = hashlib.sha256(frames["post_112"].tobytes()).hexdigest()
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
        results, kept, preds = {}, {}, {}

        def single(name, x):
            clock.check(f"readout {name}")
            g, diag = ic.gram(x)
            results[name], kept[name], preds[name] = BASE.evaluate_source(
                name, g, diag, xy, dx, dy, fold, priors, masks, train_mask, split_priors
            )
            results[name]["feature_dim"] = int(x.shape[1])

        # --- stage 1: anchors, before any training ---
        clock.check("anchors")
        features, _digests059 = BASE.encoder_features(
            {"post_112": frames["post_112"], "post_112_hidden": frames["post_112_hidden"]},
            task059,
        )
        flat = lambda a: a.reshape(len(a), -1) / 255.0  # noqa: E731
        single("L_raw", flat(frames["post_112"]))
        single("L_E0", features[("E0", "post_112")])
        single("L_random", features[("random", "post_112")])
        if not smoke:
            comparison = es.anchor_compare(
                anchor_reference(json.loads(TASK061_REPORT.read_text())),
                {s: recomputed_view(results[s], preds[s]) for s in es.ANCHOR_SOURCES},
            )
            report["anchor"] = comparison
            es.check_anchor(comparison)
        else:
            report["anchor"] = {"verdict": "skipped (smoke)"}
        config, store, settings = open_config()
        backend = config.backend
        refs, digests = reference_models(config, store, task061)
        report["encoder_weights_sha256"] = digests
        ref_feats = {
            (name, key): stage_features(model, frames[key])
            for name, model in refs.items()
            for key in ("post_112", "post_112_hidden")
        }
        for name in ("E0", "random"):  # the stage code must equal TASK-061's feature path
            label = "L_E0" if name == "E0" else "L_random"
            same = np.array_equal(
                ref_feats[(name, "post_112")]["probe"], features[(name, "post_112")]
            )
            report.setdefault("stage_code_equals_task061_features", {})[label] = bool(same)
            if not same:
                raise ic.GuardError(f"G-anchor: stage features of {name} differ from TASK-061's")
        single("E0_tok", ref_feats[("E0", "post_112")]["tokens"])
        single("F_tok", ref_feats[("random", "post_112")]["tokens"])
        if digests["random"] != es.SEED0_DIGEST[True]:  # G-init for F-cls and F-tok
            raise es.GuardError("G-init: the random reference is not the seed-0 init")
        report["F_cls_is_L_random"] = True
        plain = build_model(backend, store, settings, "PLAIN", "cpu", {})
        report["F_plain_init_digest"] = es.check_init(plain, readout_heads=False)  # G-init
        plain.eval()
        single("F_plain", stage_features(plain, frames["post_112"])["probe"])
        del plain
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
        del refs
        # --- stage 2-4: training, reload, features, diagnostics ---
        encoders_dir = output / "encoders"
        steps = smoke_steps if smoke else es.STEPS
        cap = 600.0 if smoke else es.PER_ENCODER_SECONDS
        trained, feats, diagnostics = {}, {}, {}
        for half in es.HALVES:
            clock.check(f"decode half {half}")
            arrays = load_half(store, episodes_by_half[half], check_budget=lambda: None)
            report.setdefault("training_data", {})[half] = {
                "episodes": len(arrays.episode_ids),
                "observations": int(len(arrays.states)),
                "episode_ids_sha256": hashlib.sha256(
                    json.dumps(sorted(arrays.episode_ids)).encode()
                ).hexdigest(),
            }
            for name in es.MODELS:
                record = train_encoder(
                    name,
                    half,
                    arrays,
                    store,
                    settings,
                    backend,
                    encoders_dir,
                    clock,
                    steps=steps,
                    device=device,
                    cap_seconds=cap,
                    source=report["source"],
                )
                trained[(name, half)] = record
                if record["status"] != "completed":
                    continue  # a failed arm is demoted; its weights are never read
                model = reload_cpu(record, backend, store, settings)
                feats[(name, half)] = {
                    key: stage_features(model, frames[key])
                    for key in ("post_112", "post_112_hidden")
                }
                stages = {
                    k: CAL.encoder_stages(model, cal_frames[k])
                    for k in ("post", "apple_hidden", "plate_hidden")
                }
                own = es.half_of_fold(fold) == half
                diagnostics[(name, half)] = {
                    "collapse_training_half": collapse_statistics(model, arrays),
                    "collapse_post_look": {
                        k: _stats(v) for k, v in feats[(name, half)]["post_112"].items()
                    },
                    "action_sensitivity": action_sensitivity(model, arrays, store),
                    "label_free_own_held_out_roots": {
                        stage: CAL.shares(
                            stages["post"][stage][own],
                            stages["apple_hidden"][stage][own],
                            stages["plate_hidden"][stage][own],
                        )
                        for stage in ("final_tokens", "cls", "projector", "probe_feature")
                    },
                }
                del model
            del arrays
        report["encoders"] = {f"{k[0]}_{k[1]}": v for k, v in trained.items()}
        report["diagnostics"] = {f"{k[0]}_{k[1]}": v for k, v in diagnostics.items()}
        report["label_free_reference"] = label_free
        # --- stage 5: cross-fitted readouts ---
        halves = es.half_of_fold(fold)

        def crossfit(name, model_name, kind):
            clock.check(f"readout {name}")
            if any(not trained[(model_name, h)]["status"] == "completed" for h in es.HALVES):
                return False
            grams = {h: ic.gram(feats[(model_name, h)]["post_112"][kind]) for h in es.HALVES}
            results[name], kept[name], preds[name] = es.evaluate_crossfit_source(
                grams, xy, dx, dy, fold, priors, masks, train_mask, split_priors
            )
            results[name]["feature_dim"] = int(feats[(model_name, "A")]["post_112"][kind].shape[1])
            return True

        available = {
            "R0_cls": crossfit("R0_cls", "R0", "probe"),
            "A_tok": crossfit("A_tok", "R0", "tokens"),
            "A_sig": crossfit("A_sig", "SIG", "probe"),
            "A_rec": crossfit("A_rec", "REC", "probe"),
            "A_plain": crossfit("A_plain", "PLAIN", "probe"),
        }
        source_of = {"A-tok": "A_tok", "A-sig": "A_sig", "A-rec": "A_rec", "A-plain": "A_plain"}
        floor_source = {"F-tok": "F_tok", "F-cls": "L_random", "F-plain": "F_plain"}
        arms, pvalues = {}, {}
        for arm in es.ARMS:
            name = source_of[arm]
            model_name = es.ARM_MODEL[arm]
            floor = floor_source[es.ARM_FLOOR[arm]]
            entry = {
                "source": name,
                "floor": floor,
                "trained": all(
                    trained[(model_name, h)]["status"] == "completed" for h in es.HALVES
                ),
                "encoders": {h: trained[(model_name, h)]["status"] for h in es.HALVES},
            }
            entry["not_collapsed"] = entry["trained"] and all(
                es.collapse_ok(diagnostics[(model_name, h)]["collapse_training_half"]["probe"])
                for h in es.HALVES
            )
            entry["state"] = es.arm_state(
                trained=entry["trained"], not_collapsed=entry["not_collapsed"]
            )
            if available[name]:
                entry["succeeds"] = bool(results[name]["all"]["succeeds"])
                entry["beats_prior"] = bool(results[name]["all"]["beats_prior"])
                entry["floor_comparison"] = ic.beats_random_floor(
                    kept[name]["all"], kept[floor]["all"]
                )
                entry["beats_floor"] = bool(entry["floor_comparison"]["beats_random_floor"])
                entry["pvalues"] = es.arm_pvalues(
                    preds[name]["xy"],
                    preds[name]["dx"],
                    xy,
                    dx,
                    priors,
                    kept[floor]["all"],
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
        holm = es.holm(pvalues)
        for arm in es.ARMS:
            name, entry = source_of[arm], arms[arm]
            model_name, kind = es.ARM_MODEL[arm], es.ARM_FEATURE[arm]
            reproduces = ablation_ok == n
            hidden_eval = None
            if available[name] and reproduces:
                hidden_grams = {
                    h: ic.cross_gram(
                        feats[(model_name, h)]["post_112_hidden"][kind],
                        feats[(model_name, h)]["post_112"][kind],
                    )
                    for h in es.HALVES
                }
                xy_hat = es.crossfit_predict_hidden(preds[name]["xy_fits"], fold, hidden_grams)
                dx_hat = es.crossfit_predict_hidden(preds[name]["dx_fits"], fold, hidden_grams)
                hidden_eval = ic.evaluate(xy_hat, dx_hat[:, 0], xy, dx, priors, np.ones(n, bool))
            entry["spurious_check"] = orp.spurious_verdict(
                hidden_eval, renderer_reproduces=reproduces
            ) | {"hidden_all_roots": ic.public(hidden_eval) if hidden_eval else None}
            entry["spurious"] = bool(entry["spurious_check"]["spurious"])
            entry["holm_rejected"] = bool(holm["rejected"][arm])
            entry["passes"] = es.passes(
                trained=entry["trained"],
                not_collapsed=entry["not_collapsed"],
                succeeds=entry["succeeds"],
                beats_floor=entry["beats_floor"],
                rejected=entry["holm_rejected"],
                spurious=entry["spurious"],
            )
            entry["qualifier"] = es.qualifier(
                passes=entry["passes"],
                succeeds=entry["succeeds"],
                beats_floor=entry["beats_floor"],
                rejected=entry["holm_rejected"],
                beats_prior=entry["beats_prior"],
            )
            entry["p"] = entry["pvalues"]["p"]
        report["arms"] = arms
        report["holm"] = holm
        decision = es.decide(void=False, arms=arms)
        decision["abandonment_clause_fires"] = es.abandonment_fires(decision["outcome"])
        report["decision"] = decision
        # --- reported: floors' own spurious check, pairwise comparisons ---
        pairs = [
            ("A_tok", "F_tok"),
            ("A_sig", "L_random"),
            ("A_rec", "L_random"),
            ("A_plain", "F_plain"),
            ("A_plain", "L_raw"),
            ("A_plain", "R0_cls"),
            ("F_plain", "L_random"),
            ("A_tok", "L_raw"),
            ("A_sig", "L_raw"),
            ("A_rec", "L_raw"),
            ("A_tok", "R0_cls"),
            ("A_sig", "R0_cls"),
            ("A_rec", "R0_cls"),
            ("A_tok", "E0_tok"),
            ("R0_cls", "L_E0"),
            ("E0_tok", "L_E0"),
            ("F_tok", "L_raw"),
        ]
        report["pairwise_reported"] = {
            f"{a}_minus_{b}": {
                s: ic.compare_sources(kept[a][s], kept[b][s])
                for s in ("all", "reset_occluded", "reset_visible")
                if s in kept.get(a, {}) and s in kept.get(b, {})
            }
            for a, b in pairs
            if a in kept and b in kept
        }
        report["results"] = results
        report["per_root"] = [
            {
                "seed": r["seed"],
                "split": r["split"],
                "fold": int(fold[i]),
                "half": str(halves[i]),
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
    except (ic.GuardError, VoidRun) as error:
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


def _stats(values) -> dict:
    import torch

    from embodied_jepa.models.base import _statistics

    return _statistics(torch.from_numpy(np.asarray(values, np.float64)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not args.smoke and (args.smoke_steps != 3 or args.device != "mps"):
        raise SystemExit("--smoke-steps and --device are smoke-only overrides")
    report = run(args.output, smoke=args.smoke, smoke_steps=args.smoke_steps, device=args.device)
    print(json.dumps({k: report.get(k) for k in ("status", "decision", "elapsed_seconds")}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
