"""TASK-064 readability gate (``apple_look_corpus_v1`` §7): TASK-063's P-cls on the corpus.

On the 190 train + val roots of the sealed look-prefix corpus -- **never the test split** -- this
reads each root episode's stored post-look frame (frame 8), featurises it with the pinned frozen
DINOv2 ViT-S/14 exactly as TASK-063 did (``pretrained_encoder``: bicubic 112 -> 224 px, no crop,
ImageNet normalisation, CLS after the final LayerNorm), and runs TASK-061's readouts, bars and
TASK-063's floor rule against the same architecture's seed-0 random init (R-cls). The apple-hidden
check uses a fresh re-render of the same post-look state.

Guards (a failure voids the run): Q-split (exactly the planned train + val roots), Q-render (the
stored reset and post-look frames equal a fresh re-render on every root), Q-expert (the recorded
first policy base action equals the recomputed post-look expert on the non-aim roots, and the
stored apple label equals the reset truth), Q-folds (the pinned fold hash), G-weights (TASK-063's
pinned files and digests), G-repro (a second forward pass is bit-identical), and the global wall
cap.

The pieces are TASK-059/061/062/063's, loaded or imported unchanged: ``info_ceiling``,
``observation_reprobe``, ``encoder_study.arm_pvalues``, ``pretrained_encoder`` and
``scripts/probe_info_ceiling.py`` (``evaluate_source``, ``Renders``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import encoder_study as es  # noqa: E402
from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa import look_corpus as lc  # noqa: E402
from embodied_jepa import observation_reprobe as orp  # noqa: E402
from embodied_jepa import pretrained_encoder as pe  # noqa: E402
from embodied_jepa.contracts import ContractError  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-look-corpus-v1.json"
TASK063_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-pretrained-encoder-v1.json"
FREE = [6, 7, 8, 9, 10, 11, 13]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load("_probe_info_ceiling", "scripts/probe_info_ceiling.py")


class Clock:
    def __init__(self, start_elapsed: float, cap: float = lc.GLOBAL_WALL_SECONDS):
        self.start, self.offset, self.cap = time.monotonic(), float(start_elapsed), cap

    def elapsed(self) -> float:
        return self.offset + time.monotonic() - self.start

    def check(self, stage: str) -> None:
        if self.elapsed() > self.cap:
            raise lc.GuardError(f"global wall cap {self.cap} s reached before {stage}")


def frames_sha256(frames) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


class CorpusFrames:
    """Stored frames of ALLOWED (train + val root) episodes only, after the sha256 check."""

    def __init__(self, store, allowed_ids):
        self.store = store
        self.allowed = frozenset(allowed_ids)
        self.rows = {r["episode_id"]: r for r in store.manifest["episodes"]}

    def row(self, episode_id):
        if episode_id not in self.allowed:
            raise lc.GuardError(f"Q-split: refusing to open {episode_id}: not a read root")
        return self.rows[episode_id]

    def frames(self, episode_id, indices):
        row = self.row(episode_id)  # Q-split refusal before anything is imported or opened

        import pyarrow.parquet as pq
        from PIL import Image

        path = self.store.root / row["path"]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != self.store.manifest["sha256"][row["path"]]
        ):
            raise lc.GuardError(f"episode file hash mismatch: {episode_id}")
        table = pq.read_table(path, columns=["observation.images.onboard_rgb", "frame_index"])
        index = table["frame_index"].to_pylist()
        out = []
        for i in indices:
            if index[i] != i:
                raise lc.GuardError(f"stored row {i} of {episode_id} is not frame {i}")
            item = table["observation.images.onboard_rgb"][i].as_py()
            with Image.open(io.BytesIO(item["bytes"])) as image:
                out.append(np.asarray(image).copy())
        return out


# ----- guards, as testable helpers (protocol §9) -------------------------------------------------
def check_folds(got: str, want: str) -> None:
    if got != want:
        raise lc.GuardError(f"Q-folds: fold hash {got[:12]} != pinned {want[:12]}")


def check_weights(weights: dict, task063: dict) -> None:
    if weights["pretrained_digest"] != task063["encoder"]["pretrained_weights_digest"]:
        raise lc.GuardError("G-weights: pretrained digest differs from TASK-063's pin")
    if weights["floor_digest"] != task063["floors"]["weights_digest"]:
        raise lc.GuardError("G-weights: floor digest differs from TASK-063's pin")


def check_repro(first: dict, again: dict) -> None:
    for key in again:
        if key not in first or not np.array_equal(again[key], first[key]):
            raise lc.GuardError(f"G-repro: second forward pass differs at {key}")


def finite_features(featurise, name: str, frames):
    """G-finite: a non-finite feature is mechanical, not a readability result -- the run is V."""
    try:
        return featurise(frames)
    except ContractError as error:
        if "non-finite" not in str(error):
            raise
        raise lc.GuardError(f"G-finite: non-finite {name} features: {error}") from error


def recorded_first_policy_action(labels, episode_id):
    """Q-expert: the root must have stored its first policy command after the look."""
    base = np.asarray(labels["collector__base_action"])
    if len(base) <= lc.LOOK_STEPS:
        raise lc.GuardError(
            f"Q-expert: {episode_id} stored no policy command after the look ({len(base)} steps)"
        )
    return base[lc.LOOK_STEPS]


def warm(renders, data) -> None:
    """Every renderer renders once and discards the result (TASK-061 §5)."""
    for renderer in (*renders.seg.values(), renders.native, renders.small):
        renderer.update_scene(data, camera="onboard_rgb")
        renderer.render()


def measure(roots, corpus, *, clock):
    """Re-simulate every read root's reset and look; compare with the stored frames."""
    from embodied_jepa import readout_labels
    from embodied_jepa.scripted import apple_collector_policy

    collector = _load("_collect_apple_look", "scripts/collect_apple_look.py")
    robot = collector.make_robot()
    renders = BASE.Renders(robot)
    warm(renders, robot.sim.data)
    rows = []
    out = {k: [] for k in ("stored_reset", "stored_post", "post", "hidden")}
    try:
        for root in roots:
            clock.check("readability measurement")
            truth = robot.reset(
                seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"]
            )
            reset_frame = robot.observe().images["onboard_rgb"][0].copy()
            stored_reset, stored_post = corpus.frames(root["episode_id"], (0, lc.DECISION_FRAME))
            labels = readout_labels.load_privileged(
                corpus.store.root,
                corpus.row(root["episode_id"]),
                acknowledge_privileged_training_labels=True,
            )
            reset_pixels = renders.apple_pixels(lc.IMAGE_SIZE)
            during = []

            def after_step(_step, sim=robot.sim, start=truth, during=during):
                now = sim.task_truth()
                during.append(
                    (
                        float(
                            np.abs(now["object_position"][:2] - start["object_position"][:2]).max()
                        ),
                        bool(now["hand_contact"]),
                    )
                )

            orp.execute_look(robot, after_step)
            post = robot.observe().images["onboard_rgb"][0].copy()
            rerender = robot.sim.render().copy()
            expert_post = apple_collector_policy(truth).action(robot)
            post_pixels = renders.apple_pixels(lc.IMAGE_SIZE)
            plain = renders.small_rgb()
            hidden = renders.small_rgb(hide_apple=True)
            rows.append(
                {
                    "seed": int(root["seed"]),
                    "split": root["split"],
                    "episode_id": root["episode_id"],
                    "aim_offset": root["aim_offset_xy_m"] is not None,
                    "apple_xy_truth": [float(v) for v in truth["object_position"][:2]],
                    "apple_xy_label": [
                        float(v) for v in labels["privileged__apple_position_world"][0][:2]
                    ],
                    "expert_post_look": [float(v) for v in expert_post[FREE]],
                    "recorded_post_look": [
                        float(v)
                        for v in recorded_first_policy_action(labels, root["episode_id"])[FREE]
                    ],
                    "recorded_look_phases_ok": bool(
                        np.all(
                            labels["collector__phase_index"][: lc.LOOK_STEPS] == lc.LOOK_PHASE_INDEX
                        )
                    ),
                    "apple_pixels_reset_112": int(reset_pixels),
                    "apple_pixels_post_look_112": int(post_pixels),
                    "look_apple_xy_move_max_m": max(d[0] for d in during),
                    "look_hand_contact": any(d[1] for d in during),
                    "stored_reset_equals_rerender": orp.frames_identical(stored_reset, reset_frame),
                    "stored_post_equals_rerender": orp.frames_identical(stored_post, post),
                    "post_rerender_stable": orp.frames_identical(post, rerender),
                    "ablation_reproduces": orp.frames_identical(plain, post),
                }
            )
            out["stored_reset"].append(stored_reset)
            out["stored_post"].append(stored_post)
            out["post"].append(post)
            out["hidden"].append(hidden)
    finally:
        renders.close()
        robot.close()
    return rows, {k: np.stack(v) for k, v in out.items()}


def readability(dataset, plan, *, clock_start: float = 0.0, smoke: bool = False) -> dict:
    """The §7 gate. Returns a JSON-able dict with ``passes``. Raises ``lc.GuardError`` (or
    ``pe.WeightsError``, ``ic.GuardError``) on a guard failure: the caller records V."""
    import torch

    from embodied_jepa.data import DatasetStore
    from embodied_jepa.training import peak_rss_bytes

    clock = Clock(clock_start)
    begin = time.monotonic()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else None
    task063 = json.loads(TASK063_MANIFEST.read_text())
    store = DatasetStore(dataset)  # verifies every recorded hash (bytes only)
    roots = lc.read_roots(plan)
    splits_by_episode = {e: name for name, ids in store.manifest["splits"].items() for e in ids}
    if not smoke:
        lc.check_read_split(roots, splits_by_episode)
    corpus = CorpusFrames(store, [r["episode_id"] for r in roots])
    seeds = [r["seed"] for r in roots]
    fold = ic.fold_of(len(roots))
    fold_hash = ic.fold_assignment_sha256(seeds, fold)
    if not smoke and manifest is not None:
        check_folds(fold_hash, manifest["readability_gate"]["fold_assignment_sha256"])
    gate = {
        "roots": len(roots),
        "splits": {k: sum(r["split"] == k for r in roots) for k in lc.READ_SPLITS},
        "fold_assignment_sha256": fold_hash,
        "test_split_decoded": False,
    }
    # --- frames, guards Q-render and Q-expert ---
    rows, frames = measure(roots, corpus, clock=clock)
    n = len(rows)
    lc.check_render([r["stored_reset_equals_rerender"] for r in rows], "stored reset frame")
    lc.check_render([r["stored_post_equals_rerender"] for r in rows], "stored post-look frame")
    lc.check_render([r["post_rerender_stable"] for r in rows], "post-look re-render")
    if not all(r["recorded_look_phases_ok"] for r in rows):
        raise lc.GuardError("Q-expert: a root's first 8 recorded phases are not the look")
    gate["expert"] = lc.check_expert(
        [r["expert_post_look"] for r in rows],
        [r["recorded_post_look"] for r in rows],
        [r["aim_offset"] for r in rows],
    )
    gate["apple_label_max_abs_m"] = lc.check_apple_label(
        [r["apple_xy_label"] for r in rows], [r["apple_xy_truth"] for r in rows]
    )
    gate["stored_post_look_frames_sha256"] = frames_sha256(frames["stored_post"])
    ablation_ok = int(sum(r["ablation_reproduces"] for r in rows))
    gate["ablation_renderer_reproduces"] = ablation_ok
    xy = np.array([r["apple_xy_truth"] for r in rows], np.float64)
    dx = np.array([r["expert_post_look"][0] for r in rows], np.float64)
    dy = np.array([r["expert_post_look"][1] for r in rows], np.float64)
    if smoke:
        noise = np.random.default_rng(0)
        xy = xy.mean(0) + noise.normal(0, 0.017, xy.shape)
        dx = np.clip(noise.normal(0.1, 0.4, n), -0.4, 0.4)
        dx[dx == 0] = 0.1
        dy = noise.normal(-0.39, 0.02, n)
        gate["smoke_targets"] = "seeded noise; no real target reached a readout"
    train_mask = np.array([r["split"] == "train" for r in rows])
    occluded = np.array([r["apple_pixels_reset_112"] == 0 for r in rows])
    priors = ic.prior_predictions(xy, dx, occluded, fold, dy=dy)
    split_priors = ic.prior_predictions(xy, dx, occluded, np.where(train_mask, 1, 0), dy=dy)
    masks = {
        k: m
        for k, m in {
            "all": np.ones(n, bool),
            "reset_occluded": occluded,
            "reset_visible": ~occluded,
        }.items()
        if m.any()
    }
    gate["prior_summary"] = ic.prior_summary(xy, dx, occluded, fold, train_mask)
    gate["visibility"] = {
        "reset_occluded_roots": int(occluded.sum()),
        "post_look_occluded_roots": int(sum(r["apple_pixels_post_look_112"] == 0 for r in rows)),
        "post_look_apple_pixels_quantiles_min_p25_p50_p75_max": np.percentile(
            [r["apple_pixels_post_look_112"] for r in rows], [0, 25, 50, 75, 100]
        ).tolist(),
        "look_apple_xy_move_max_m": max(r["look_apple_xy_move_max_m"] for r in rows),
        "look_hand_contact_roots": int(sum(r["look_hand_contact"] for r in rows)),
    }
    # --- G-weights, features, G-repro ---
    clock.check("weights")
    pretrained = pe.load_pretrained(pe.DEFAULT_DIRECTORY)
    floor = pe.random_init(pe.DEFAULT_DIRECTORY)
    weights = {
        "files_sha256": pe.check_files(pe.DEFAULT_DIRECTORY),
        "pretrained_digest": pe.weights_digest(pretrained),
        "floor_digest": pe.weights_digest(floor),
    }
    check_weights(weights, task063)
    gate["weights"] = weights
    gate["environment"] = {"torch": torch.__version__, "torch_threads": torch.get_num_threads()}
    feats, finite = {}, True
    for name, model in (("P", pretrained), ("R", floor)):
        clock.check(f"features {name}")
        feats[name] = {
            key: finite_features(lambda x, m=model: pe.features(m, x), name, frames[source])
            for key, source in (("post", "stored_post"), ("hidden", "hidden"))
        }
        for key in ("post", "hidden"):
            tokens = feats[name][key]["tokens"]
            feats[name][key]["mean"] = tokens.reshape(n, pe.GRID * pe.GRID, pe.WIDTH).mean(1)
    if "P" in feats:
        check_repro(feats["P"]["post"], pe.features(pretrained, frames["stored_post"]))
        gate["G_repro"] = "bit-identical"
    # --- readouts ---
    results, kept, preds, feature_sha, seconds = {}, {}, {}, {}, {}

    def single(name, x):
        clock.check(f"readout {name}")
        start = time.monotonic()
        x = np.asarray(x, np.float64).reshape(n, -1)
        feature_sha[name] = frames_sha256(x)
        g, diag = ic.gram(x)
        results[name], kept[name], preds[name] = BASE.evaluate_source(
            name, g, diag, xy, dx, dy, fold, priors, masks, train_mask, split_priors
        )
        results[name]["feature_dim"] = int(x.shape[1])
        seconds[name] = time.monotonic() - start

    for enc in ("P", "R"):
        if enc in feats:
            for point, label in (("cls", "cls"), ("tokens", "tok"), ("mean", "mean")):
                single(f"{enc}_{label}", feats[enc]["post"][point])
    single("L_raw", frames["stored_post"].reshape(n, -1) / 255.0)
    # --- the gate: P-cls against R-cls ---
    arm = {"source": "P_cls", "floor": "R_cls", "finite": finite}
    if finite:
        arm["succeeds"] = bool(results["P_cls"]["all"]["succeeds"])
        arm["beats_prior"] = bool(results["P_cls"]["all"]["beats_prior"])
        arm["floor_comparison"] = ic.beats_random_floor(kept["P_cls"]["all"], kept["R_cls"]["all"])
        arm["beats_floor"] = bool(arm["floor_comparison"]["beats_random_floor"])
        arm["pvalues"] = es.arm_pvalues(
            preds["P_cls"]["xy"],
            preds["P_cls"]["dx"],
            xy,
            dx,
            priors,
            kept["R_cls"]["all"],
            kept["P_cls"]["all"],
        )
        hidden_eval = None
        if ablation_ok == n:
            g_new, norms = ic.cross_gram(feats["P"]["hidden"]["cls"], feats["P"]["post"]["cls"])
            xy_hat = orp.predict_with_fold_readouts(preds["P_cls"]["xy_fits"], fold, g_new, norms)
            dx_hat = orp.predict_with_fold_readouts(preds["P_cls"]["dx_fits"], fold, g_new, norms)
            hidden_eval = ic.evaluate(xy_hat, dx_hat[:, 0], xy, dx, priors, np.ones(n, bool))
        arm["spurious_check"] = orp.spurious_verdict(
            hidden_eval, renderer_reproduces=ablation_ok == n
        ) | {"hidden_all_roots": ic.public(hidden_eval) if hidden_eval else None}
        arm["spurious"] = bool(arm["spurious_check"]["spurious"])
        arm["p"] = float(arm["pvalues"]["p"])
    else:
        arm |= {"succeeds": False, "beats_floor": False, "spurious": True, "p": 1.0}
    arm["passes"] = lc.readability_passes(
        succeeds=arm["succeeds"],
        beats_floor=arm["beats_floor"],
        p_arm=arm["p"],
        spurious=arm["spurious"],
        finite=finite,
    )
    gate["arm"] = arm
    gate["passes"] = bool(arm["passes"])
    pairs = [("P_cls", "R_cls"), ("P_cls", "L_raw"), ("P_tok", "R_tok"), ("P_tok", "L_raw")]
    gate["pairwise_reported"] = {
        f"{a}_minus_{b}": {
            s: ic.compare_sources(kept[a][s], kept[b][s])
            for s in masks
            if s in kept[a] and s in kept[b]
        }
        for a, b in pairs
        if a in kept and b in kept
    }
    gate["results"] = results
    gate["feature_sha256"] = feature_sha
    gate["readout_seconds"] = seconds
    gate["per_root"] = [
        {
            "seed": r["seed"],
            "split": r["split"],
            "fold": int(fold[i]),
            "reset_occluded": bool(occluded[i]),
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
    gate["seconds"] = time.monotonic() - begin
    gate["peak_rss_bytes"] = peak_rss_bytes()
    clock.check("readability report")
    return gate
