"""TASK-073 offline pieces: featurisation, readouts, W/N training, and the O1-O4 statistics.

Protocol ``docs/experiments/apple_wm_critic_v2.md`` §5-§6. Stored frames only; nothing here
simulates. NumPy at import; torch, transformers and the frozen-token model are imported by the
functions that need them.

* **Features.** Frozen DINOv2 ViT-S/14 final patch tokens, pooled to 4 x 4 (``pool_tokens``), for
  the frames of each root inside ``FEATURE_BAND``; computed on CUDA (owner D6) and checked against
  the CPU path on an anchor subset (max |difference| of the pooled latent <= 1e-3).
* **W and N.** TASK-066's LeWM token predictor, configuration unchanged
  (``token_dynamics.MODEL_CONFIG``), trained on 17-frame windows (h = 16) inside the band; N is
  given zero actions in training and evaluation; W and N share each seed's window sampler.
* **Readouts.** ``R_off`` (pooled latent -> apple-minus-plate offset, m) and ``R_mid`` (full
  tokens -> plate xy, m), ``models.latent_critic.RidgeReadout``.
* **O1/O2** on the corpus's val roots; **O3/O4** on the ranking groups of cohort R.
"""

from __future__ import annotations

import time

import numpy as np

from embodied_jepa import latent_dynamics as ld
from embodied_jepa import token_dynamics as td
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError

EVAL_CHUNK = 256  # memory: the O1 and val-criterion evaluation batch (protocol §15.6)
FEATURE_BATCH = 128


# ----- features ---------------------------------------------------------------------------------
def pooled_features(encoder, frames, *, device: str = "cpu", batch: int = FEATURE_BATCH):
    """uint8 [N, 112, 112, 3] -> pooled 4 x 4 latents float32 [N, 6144] (and nothing else)."""
    import torch

    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    frames = np.asarray(frames)
    out = []
    encoder.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch):
            pixels = pe.preprocess(frames[start : start + batch]).to(device)
            last = encoder(pixel_values=pixels).last_hidden_state[:, 1:]
            tokens = last.reshape(len(last), -1).float().cpu().numpy()
            out.append(pool_tokens(tokens.astype(np.float64), wc.W_TOKEN_GRID))
    encoder.to("cpu")
    result = np.concatenate(out) if out else np.zeros((0, 6144), np.float32)
    if not np.isfinite(result).all():
        raise wc.GuardError("G-finite: a pooled feature is not finite")
    return result


def full_tokens_cpu(encoder, frames) -> np.ndarray:
    """The P readout's feature path (``first_policy_perception.featurise``: CPU, batch size 1):
    full tokens [N, 98304], stored as float32 (memory; protocol §15). R-mid is fitted on them
    and reads the same path in the closed loop."""
    from embodied_jepa import first_policy_perception as fpp

    frames = np.asarray(frames)
    out = np.empty((len(frames), 256 * 384), np.float32)
    for i in range(len(frames)):
        out[i] = fpp.featurise(encoder, frames[i : i + 1])[0]
    return out


def anchor_check(encoder, frames, gpu_pooled) -> dict:
    """G-anchor: the CUDA pooled features of ``frames`` against the closed loop's CPU path
    (``hybrid_selection.encode_frame``: batch size 1)."""
    from embodied_jepa.hybrid_selection import encode_frame

    cpu = np.stack([encode_frame(encoder, f)[1] for f in np.asarray(frames)])
    diff = float(np.abs(cpu.astype(np.float64) - np.asarray(gpu_pooled, np.float64)).max())
    bound = wc.FEATURE_ANCHOR["max_abs_pooled_difference"]
    if not diff <= bound:
        raise wc.GuardError(f"G-anchor: CUDA and CPU pooled features differ by {diff} > {bound}")
    return {"frames": int(len(cpu)), "max_abs_difference": diff, "bound": bound}


# ----- the feature table ------------------------------------------------------------------------
class Table:
    """Band frames of many roots in one array: ``features`` [F, 6144] float32, ``actions``
    [F, 14] (the applied command leaving each frame; the last frame of a root has none),
    ``offset_m`` [F, 2] and ``plate_m`` [F, 2] (privileged labels), per-frame ``root`` and
    ``step``; ``roots`` lists dicts with ``id``, ``split``, ``source``, ``first``, ``length``,
    ``shift_step``."""

    def __init__(self, capacity: int | None = None, width: int = 6144):
        """With ``capacity`` (frames), the feature rows are written straight into one lazily
        committed array (memory; protocol §15.6): no part list and no second copy at ``seal``."""
        self.parts: dict[str, list] = {k: [] for k in ("features", "actions", "offset", "plate")}
        self.roots: list[dict] = []
        self.size = 0
        self._features = None if capacity is None else np.empty((int(capacity), width), np.float32)

    def add(self, root: dict, features, actions, apple, plate, first_step: int) -> None:
        n = len(features)
        if not (len(actions) == len(apple) == len(plate) == n):
            raise ContractError("a root's band arrays disagree in length")
        if self._features is not None:
            if self.size + n > len(self._features):
                raise ContractError("the table's capacity is exceeded")
            self._features[self.size : self.size + n] = features
        else:
            self.parts["features"].append(np.asarray(features, np.float32))
        self.parts["actions"].append(np.asarray(actions, np.float32))
        self.parts["offset"].append(
            np.asarray(apple, np.float64)[:, :2] - np.asarray(plate, np.float64)[:, :2]
        )
        self.parts["plate"].append(np.asarray(plate, np.float64)[:, :2])
        self.roots.append(root | {"first": self.size, "length": n, "first_step": int(first_step)})
        self.size += n

    def seal(self) -> None:
        """Join the parts without holding two copies (memory; protocol §15.6): the output is
        allocated lazily and each part is released as soon as it is copied."""
        if self._features is not None:
            self.features = self._features[: self.size]  # a view: untouched rows stay uncommitted
            self._features = None
            self.parts.pop("features")
        for key in list(self.parts):
            parts = self.parts[key]
            out = np.empty((sum(len(p) for p in parts), *parts[0].shape[1:]), parts[0].dtype)
            at = 0
            while parts:
                part = parts.pop(0)
                out[at : at + len(part)] = part
                at += len(part)
                del part
            setattr(self, key, out)
        self.parts = {}

    def windows(self, root_indices, *, horizon=16, stride=1, start_min=None, start_steps=None):
        """Global start rows of windows (horizon + 1 frames) inside each listed root."""
        starts, owners = [], []
        for i in root_indices:
            r = self.roots[i]
            last = r["length"] - horizon - 1  # frames 0..length-1; the window needs +horizon
            for k in range(0, last + 1, stride):
                step = r["first_step"] + k
                if start_min is not None and step < start_min:
                    continue
                if start_steps is not None and step not in start_steps:
                    continue
                starts.append(r["first"] + k)
                owners.append(i)
        return np.asarray(starts, np.int64), np.asarray(owners, np.int64)

    def gather(self, starts, horizon=16):
        steps = np.arange(horizon + 1)
        return self.features[starts[:, None] + steps], self.actions[starts[:, None] + steps[:-1]]


# ----- readouts ---------------------------------------------------------------------------------
def fit_r_off(table: Table, train_roots) -> tuple:
    """R_off on the train roots' band frames at steps 304-592 every 16 (primal ridge)."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    steps = set(range(wc.DECISION_STEPS[0], wc.DECISION_STEPS[-1] + wc.CHUNK + 1, wc.CHUNK))
    rows, groups = [], []
    for i in train_roots:
        r = table.roots[i]
        for k in range(r["length"]):
            if r["first_step"] + k in steps:
                rows.append(r["first"] + k)
                groups.append(r["id"])
    rows = np.asarray(rows)
    readout = RidgeReadout.fit(
        table.features[rows],
        table.offset[rows],
        groups,
        lambdas=wc.READOUTS["lambda_grid_relative"],
        folds=wc.READOUTS["inner_folds"],
        seed=wc.READOUTS["fold_seed"],
    )
    return readout, rows


def fit_r_mid(full_tokens, plate_xy, groups):
    """R_mid (dual ridge) on full tokens at the train roots' decision frames."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    return RidgeReadout.fit(
        full_tokens,
        plate_xy,
        groups,
        lambdas=wc.READOUTS["lambda_grid_relative"],
        folds=wc.READOUTS["inner_folds"],
        seed=wc.READOUTS["fold_seed"],
        dual=True,
    )


# ----- W and N ----------------------------------------------------------------------------------
def new_model(schema, arm: str, seed: int, device: str, metadata: dict):
    from embodied_jepa.models.frozen_tokens import frozen_token_model

    return frozen_token_model(td.BACKEND)(
        schema,
        device=device,
        seed=int(seed),
        config=td.MODEL_CONFIG,
        metadata=dict(metadata) | {"protocol": wc.PROTOCOL, "arm": arm},
    )


def val_criterion(model, table: Table, starts, scale, arm: str) -> float:
    total, count = 0.0, 0
    for lo in range(0, len(starts), EVAL_CHUNK):
        f, a = table.gather(starts[lo : lo + EVAL_CHUNK])
        if arm == "N":
            a = np.zeros_like(a)
        predicted = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
        err = ld.normalized_sq_error(predicted, f[:, 1:], scale)
        total += float(err.sum())
        count += err.size
    value = total / count
    if not np.isfinite(value):
        raise wc.GuardError("G-finite: non-finite val criterion")
    return value


def sampler(seed: int) -> np.random.Generator:
    """Shared by W and N of the same seed, as TASK-065/066."""
    return np.random.default_rng(np.random.SeedSequence([wc.SEEDS["sampler_salt"], int(seed)]))


def train_model(ctx: dict, arm: str, seed: int, updates: int, select_every: int, path=None):
    """Train one model on the train windows, select on the val criterion, optionally save.

    ``ctx``: ``table``, ``train_starts``, ``val_starts``, ``scale``, ``mean``, ``std``,
    ``train_ids``, ``schema``, ``device``, ``metadata``, ``cap_seconds``."""
    started = time.monotonic()
    table = ctx["table"]
    model = new_model(ctx["schema"], arm, seed, ctx["device"], ctx["metadata"])
    model.fit_frozen_feature_normalization(
        ctx["mean"], ctx["std"], training_episode_ids=ctx["train_ids"]
    )
    rng = sampler(seed)
    starts = ctx["train_starts"]
    curve, losses, best = [], [], (np.inf, 0, None)
    for step in range(1, updates + 1):
        pick = starts[rng.integers(0, len(starts), wc.W_MODEL["batch_size"])]
        f, a = table.gather(pick)
        if arm == "N":
            a = np.zeros_like(a)
        metrics = model.train_step_features(f, np.ascontiguousarray(a, np.float32))
        if not np.isfinite(metrics["loss"]):
            raise wc.GuardError("G-finite: non-finite training loss")
        if step % 100 == 0:
            losses.append([step, float(metrics["loss"])])
            if time.monotonic() - started > ctx["cap_seconds"]:
                raise wc.GuardError(f"G-cap: model {arm}-{seed} exceeded its cap")
        if step % select_every == 0:
            value = val_criterion(model, table, ctx["val_starts"], ctx["scale"], arm)
            curve.append([step, value])
            if value < best[0]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best = (value, step, state)
    model.load_state_dict({k: v.to(ctx["device"]) for k, v in best[2].items()})
    points = [u for u, _ in curve]
    record = {
        "arm": arm,
        "seed": int(seed),
        "updates": int(updates),
        "select_every": int(select_every),
        "val_curve": curve,
        "losses": losses,
        "selected_update": int(best[1]),
        "selected_val_criterion": float(best[0]),
        "selected_in_last_two_points": bool(best[1] in points[-2:]),
        "seconds": time.monotonic() - started,
        "implementation_sha256": model.implementation_sha256,
        "frozen_encoder_digest": model.frozen_encoder_digest,
    }
    if path is not None:
        model.save(path)
    return model, record


def weights_sha256(model) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# ----- O1 ---------------------------------------------------------------------------------------
def _ratio(num, den, clusters, seed=wc.SEEDS["bootstrap"]):
    labels = np.unique(clusters)
    idx = ld.cluster_bootstrap_indices(len(labels), seed=seed, resamples=wc.BOOTSTRAP_RESAMPLES)
    return ld.cluster_ratio(num, den, clusters, idx)


def o1_statistics(models: dict, n_models: dict, table: Table, windows: dict, scale, basis):
    """O1's quantities for every W seed. ``windows[name] = (starts, owners)`` for "overall" and
    "post_shift"; ``models[seed]`` / ``n_models[seed]`` are W and N."""
    out = {}
    wrong_rng = np.random.default_rng(wc.SEEDS["wrong_actions"])
    perms = {name: wrong_rng.permutation(len(s)) for name, (s, _o) in windows.items()}
    for seed, model in models.items():
        per = {}
        for name, (starts, owners) in windows.items():
            clusters = np.asarray([table.roots[i]["id"] for i in owners])
            preds = {}
            for mode, m in (
                ("true", model),
                ("zero", model),
                ("wrong", model),
                ("n", n_models[seed]),
            ):
                chunks = []
                for lo in range(0, len(starts), EVAL_CHUNK):
                    sl = slice(lo, lo + EVAL_CHUNK)
                    f, a = table.gather(starts[sl])
                    if mode in ("zero", "n"):
                        a = np.zeros_like(a)
                    elif mode == "wrong":
                        a = table.gather(starts[perms[name][sl]])[1]
                    p = m.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
                    chunks.append(p[:, [h - 1 for h in wc.O1["horizons"]]])  # memory: 2 of 16
                preds[mode] = np.concatenate(chunks)
            keep = [0, *wc.O1["horizons"]]  # the start frame and the gated targets only
            sel = np.empty((len(starts), len(keep), table.features.shape[1]), np.float32)
            for lo in range(0, len(starts), EVAL_CHUNK):
                sel[lo : lo + EVAL_CHUNK] = table.gather(starts[lo : lo + EVAL_CHUNK])[0][:, keep]
            f_all = {t: sel[:, i] for i, t in enumerate(keep)}
            hi = {h: i for i, h in enumerate(wc.O1["horizons"])}
            for h in wc.O1["horizons"]:
                target = f_all[h]
                err = {
                    k: ld.normalized_sq_error(v[:, hi[h]], target, scale) for k, v in preds.items()
                }
                copy = ld.normalized_sq_error(f_all[0], target, scale)
                pred_n = preds["true"][:, hi[h]] / scale
                enc_n = target / scale
                collapse = ld.collapse_statistics(pred_n, enc_n)
                trunc = {}
                pm, em = (
                    td.Moments(np.zeros(pred_n.shape[1])),
                    td.Moments(np.zeros(pred_n.shape[1])),
                )
                pm.add(pred_n)
                em.add(enc_n)
                for k in td.TRUNCATION_CONTROLS:
                    t = td.truncated_spectrum_statistics(pm, em, k)
                    trunc[str(k)] = bool(
                        t["predicted_collapsed_fraction"] <= wc.O1["collapsed_fraction_max"]
                        and t["effective_rank_ratio"] >= wc.O1["rank_ratio_min"]
                        and t["std_ratio"] >= wc.O1["std_ratio_min"]
                    )
                shift = np.zeros(pred_n.shape[1])
                w_sm, n_sm, e_sm = (td.SessionMoments(shift, basis) for _ in range(3))
                w_sm.add(pred_n, clusters)
                n_sm.add(preds["n"][:, hi[h]] / scale, clusters)
                e_sm.add(enc_n, clusters)
                comparative = td.comparative_rank(
                    w_sm, n_sm, e_sm, seed=wc.SEEDS["comparative_rank"]
                )
                per[(name, h)] = {
                    "collapse": collapse,
                    "comparative": {"lower": comparative["ci95"][0], **comparative},
                    "truncation_pass": trunc,
                    "copy": _ratio(err["true"], copy, clusters),
                    "n": _ratio(err["true"], err["n"], clusters),
                    "shuffled": _ratio(err["wrong"], err["true"], clusters),
                    "zero": _ratio(err["zero"], err["true"], clusters),
                    "windows": int(len(starts)),
                }
        out[seed] = {"statistics": per, "gate": wc.o1_seed_passes(per)}
    return out


# ----- O2 ---------------------------------------------------------------------------------------
def o2_statistics(models: dict, n_models: dict, r_off, table: Table, starts, owners) -> dict:
    """O2 per W seed on the val windows starting at the O2 decision steps whose true offset
    moves by at least 1 cm (the moving cohort). Errors are cm distances of a readout of the
    offset at t + 16 against the true offset at t + 16."""
    moving = (
        np.linalg.norm(table.offset[starts + wc.CHUNK] - table.offset[starts], axis=1)
        >= wc.O2["moving_threshold_m"]
    )
    starts, owners = np.asarray(starts)[moving], np.asarray(owners)[moving]
    clusters = np.asarray([table.roots[i]["id"] for i in owners])
    f, a = table.gather(starts)
    truth = 100.0 * table.offset[starts + wc.CHUNK]
    step_of = np.asarray(
        [
            table.roots[i]["first_step"] + (s - table.roots[i]["first"])
            for s, i in zip(starts, owners, strict=True)
        ]
    )
    encoded = np.linalg.norm(100.0 * r_off.predict(f[:, wc.CHUNK]) - truth, axis=1)
    persistence = np.linalg.norm(100.0 * r_off.predict(f[:, 0]) - truth, axis=1)
    prior = np.linalg.norm(
        np.asarray([wc.O_STAR_CM[int(t) + wc.CHUNK] for t in step_of]) - truth, axis=1
    )
    out = {
        "baselines": {
            "persistence_median_cm": float(np.median(persistence)),
            "clock_prior_median_cm": float(np.median(prior)),
            "encoded_median_cm": float(np.median(encoded)),
            "windows": int(len(starts)),
            "roots": int(len(set(clusters))),
        }
    }
    for seed, model in models.items():
        pw = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))[:, -1]
        pn = n_models[seed].predict_features(f[:, 0], np.zeros_like(a, dtype=np.float32))[:, -1]
        ew = np.linalg.norm(100.0 * r_off.predict(pw) - truth, axis=1)
        en = np.linalg.norm(100.0 * r_off.predict(pn) - truth, axis=1)
        m = {
            "encoded_median_cm": float(np.median(encoded)),
            "predicted_median_cm": float(np.median(ew)),
            "ratio_vs_persistence": wc.cluster_median_ratio(ew, persistence, clusters),
            "n_ratio_vs_persistence": wc.cluster_median_ratio(en, persistence, clusters),
            "ratio_vs_clock_prior": wc.cluster_median_ratio(ew, prior, clusters),
        }
        out[seed] = {"statistics": m, "gate": wc.o2_passes(m)}
    return out


# ----- O3 and O4 --------------------------------------------------------------------------------
def rank_statistics(groups: list[dict], critics: dict) -> dict:
    """``groups``: every R group in cohort order (``seed``, ``point_index``, ``step``,
    ``latent``, ``chunks``, ``feasible``, ``true_costs_cm``). ``critics``: {"W0": ..., "W1": ...,
    "W2": ..., "N": ..., "copy": ...} (``LatentCritic``). Returns the blind rankers' medians and,
    per W seed, O3 and O4. L-shuf is each W seed with the start latent of reset (i + 1) mod n at
    the same point, so it is scored per seed."""
    by_point: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        by_point.setdefault(int(g["point_index"]), []).append(i)
    shuf_source = {}
    for members in by_point.values():
        for j, i in enumerate(members):
            shuf_source[i] = members[(j + 1) % len(members)]
    offsets_norm = np.linalg.norm(np.asarray(wc.CANDIDATE_OFFSETS_M), axis=1)
    true = [np.asarray(g["true_costs_cm"], np.float64) for g in groups]
    keep = [
        np.isfinite(t) & np.asarray(g["feasible"], bool) for t, g in zip(true, groups, strict=True)
    ]
    feasible = [np.asarray(g["feasible"], bool) for g in groups]  # the stand-in's, as live
    clusters = np.asarray([g["seed"] for g in groups])

    def regret_of(i, index):
        """True regret of a choice; a choice whose true outcome is not finite (the branch
        stopped) counts as the group's worst finite outcome (reviewer N3)."""
        finite = true[i][np.isfinite(true[i])]
        value = true[i][index] if np.isfinite(true[i][index]) else finite.max()
        return float(value - finite.min())

    def score(critic, latent_of):
        return [
            critic.costs(latent_of(i), groups[i]["chunks"], int(groups[i]["step"]) + wc.CHUNK)
            for i in range(len(groups))
        ]

    def rhos(scores):
        return np.asarray(
            [wc.spearman(scores[i][keep[i]], true[i][keep[i]]) for i in range(len(groups))]
        )

    own = lambda i: groups[i]["latent"]  # noqa: E731
    other = lambda i: groups[shuf_source[i]]["latent"]  # noqa: E731
    blind = {"prior-distance": rhos([offsets_norm for _ in groups])}
    if "copy" in critics:
        blind["copy-last"] = rhos(score(critics["copy"], own))
    if "N" in critics:
        blind["N"] = rhos(score(critics["N"], own))
    regret_inc = np.asarray([regret_of(i, wc.INCUMBENT_INDEX) for i in range(len(groups))])
    out = {
        "blind": {k: wc.cluster_median_ci(v, clusters) for k, v in blind.items()},
        "regret_incumbent": wc.cluster_median_ci(regret_inc, clusters),
        "groups": len(groups),
    }
    for name in [n for n in critics if n.startswith("W")]:
        scores = score(critics[name], own)
        rho = rhos(scores)
        shuf = rhos(score(critics[name], other))
        seed_blind = blind | {"L-shuf": shuf}
        best_blind = np.max(np.stack(list(seed_blind.values())), axis=0)
        # the choice uses only what the closed loop has: the costs and the stand-in's feasibility
        chosen = [wc.choose(np.where(feasible[i], scores[i], np.inf)) for i in range(len(groups))]
        regret = np.asarray([regret_of(i, chosen[i]) for i in range(len(groups))])
        o3 = {
            "rho_w": wc.cluster_median_ci(rho, clusters),
            "blind": {k: wc.cluster_median_ci(v, clusters) for k, v in seed_blind.items()},
            "margin": wc.cluster_median_ci(rho - best_blind, clusters),
        }
        o4 = {
            "regret_w": wc.cluster_median_ci(regret, clusters),
            "regret_incumbent": out["regret_incumbent"],
            "regret_difference": wc.cluster_median_ci(regret - regret_inc, clusters),
        }
        out[name] = {"O3": o3 | wc.o3_passes(o3), "O4": o4 | wc.o4_passes(o4)}
    return out
