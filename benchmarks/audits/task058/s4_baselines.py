"""Blind-baseline check for TASK-054 gates, labels only (no images, no model)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied_jepa import readout_labels
from embodied_jepa.world_model_v2 import (
    HORIZONS,
    MOVING_THRESHOLD_M,
    PHASE_CLOSE,
    PHASE_LIFT,
    _median,
    auroc,
    sibling_groups,
    window_starts,
)
from embodied_jepa.world_model_v3 import (
    RANKING_HORIZONS,
    RANKING_MIN_CANDIDATES,
    _approach_cost,
    _summarize_ranking,
)

root = Path(__import__("os").environ.get("JEPA_ROOT", ".") + "/data/apple-wide-v1")
man = json.loads((root / "meta/jepa_manifest.json").read_text())
rows = {r["episode_id"]: r for r in man["episodes"]}


def load(split):
    ids = list(man["splits"][split])
    targets, phases, lengths, rest = {}, [], [], {}
    for name in ids:
        row = rows[name]
        r = row["metadata"]["root_episode_id"]
        if r not in rest:
            rest[r] = float(
                readout_labels.load_privileged(
                    root, rows[r], acknowledge_privileged_training_labels=True
                )["privileged__apple_position_world"][0, 2]
            )
        lab = readout_labels.load_privileged(root, row, acknowledge_privileged_training_labels=True)
        t = readout_labels.targets(lab, rest[r])
        n = len(t["palm_minus_apple"])
        for k, v in t.items():
            targets.setdefault(k, []).append(v)
        ph = np.asarray(lab["collector__phase_index"], np.int16)
        p = np.empty(n, np.int16)
        p[: len(ph)] = ph
        p[len(ph) :] = ph[-1]
        phases.append(p)
        lengths.append(n)
    lengths = np.array(lengths)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return SimpleNamespace(
        episode_ids=tuple(ids),
        lengths=lengths,
        offsets=offsets,
        targets={k: np.concatenate(v) for k, v in targets.items()},
        phase=np.concatenate(phases),
        rows={n: rows[n] for n in ids},
    )


val = load("val")
train = load("train")
out = {}
win = window_starts(val, max(HORIZONS), stride=4)
starts = val.offsets[win[:, 0]] + win[:, 1]
h = 8
tg = starts + h
T = val.targets
dropped = T["apple_dropped"][:, 0] > 0.5
valid = ~dropped[starts] & ~dropped[tg]
disp = np.linalg.norm(T["palm_minus_apple"][tg] - T["palm_minus_apple"][starts], axis=1)
moving = valid & (disp >= MOVING_THRESHOLD_M)
out["windows"] = int(len(win))
out["valid"] = int(valid.sum())
out["moving"] = int(moving.sum())
out["moving_episodes"] = int(len(np.unique(win[moving, 0])))
out["true_persistence_palm_moving_median_m"] = _median(disp[moving])
out["true_persistence_palm_valid_median_m"] = _median(disp[valid])
out["frac_valid_windows_moving"] = float(moving.sum() / valid.sum())
tdrop = train.targets["apple_dropped"][:, 0] > 0.5
tm = train.targets["palm_minus_apple"][~tdrop].mean(0)
out["train_mean_palm_moving_median_m"] = _median(
    np.linalg.norm(T["palm_minus_apple"][tg] - tm, axis=1)[moving]
)
amp = T["apple_minus_plate"]
out["G3_true_persistence_median_m"] = _median(np.linalg.norm(amp[tg] - amp[starts], axis=1)[valid])
tam = train.targets["apple_minus_plate"][~tdrop].mean(0)
out["G3_train_mean_median_m"] = _median(np.linalg.norm(amp[tg] - tam, axis=1)[valid])
ph = val.phase[tg]
grasp = valid & np.isin(ph, (PHASE_CLOSE, PHASE_LIFT))
lift = valid & (ph == PHASE_LIFT)
ht = T["apple_height"][:, 0]
out["G4_grasp_windows"] = int(grasp.sum())
out["G4_zero_height_median_abs_m"] = _median(np.abs(ht[tg])[grasp])
out["G4_true_persistence_median_abs_m"] = _median(np.abs(ht[tg] - ht[starts])[grasp])
out["G4_frac_target_height_below_1cm"] = float((np.abs(ht[tg])[grasp] < 0.01).mean())
out["G4_frac_close_phase"] = float((ph[grasp] == PHASE_CLOSE).mean())
held = T["apple_held"][:, 0] > 0.5
out["G5_lift_pos"] = int(held[tg][lift].sum())
out["G5_lift_neg"] = int((~held[tg][lift]).sum())
out["G5_true_persistence_auroc"] = auroc(held[starts][lift].astype(float), held[tg][lift])
out["G5_contact_at_start_auroc"] = auroc(T["hand_contact"][starts, 0][lift], held[tg][lift])
out["G5_height_at_start_auroc"] = auroc(ht[starts][lift], held[tg][lift])
groups = sibling_groups(val)
pm = T["palm_minus_apple"]
rec = []
first_is_root = 0
for r in sorted(groups):
    mem = groups[r]
    if len(mem) < RANKING_MIN_CANDIDATES:
        continue
    preds = {}
    for name, s, stop, _ in mem:
        steps = min(max(RANKING_HORIZONS), stop - 1 - s)
        if steps < min(RANKING_HORIZONS):
            continue
        preds[name] = (s, steps)
    names = sorted(preds)
    H = 16
    usable = [n for n in names if preds[n][1] >= H and not dropped[preds[n][0] + H]]
    if len(usable) < RANKING_MIN_CANDIDATES:
        continue
    tc = _approach_cost([pm[preds[n][0] + H] for n in usable])
    first_is_root += usable[0] == r
    rec.append(
        {
            "root": r,
            "candidates": len(usable),
            "true": tc,
            "predicted": np.zeros(len(usable)),
            "shuffled": np.zeros(len(usable)),
        }
    )
s = _summarize_ranking(rec)
out["G6_constant_groups_ranked"] = s["groups_ranked"]
out["G6a_constant_spearman"] = s["within_state_spearman_pooled"]
out["G6b_constant_top1_regret_median_m"] = s["top1_regret_median_m"]
out["G6b_random_choice_regret_median_m"] = s["random_choice_regret_median_m"]
out["G6_first_sorted_is_root_groups"] = int(first_is_root)
out["G6_groups"] = len(rec)
rk = [e for e in rec if e["true"].max() - e["true"].min() >= 0.005]
out["G6_root_is_true_best_frac"] = float(np.mean([np.argmin(e["true"]) == 0 for e in rk]))
out["G6_root_regret_each_m"] = [float(e["true"][0] - e["true"].min()) for e in rk]
print(json.dumps(out, indent=1))
