import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MAIN = Path(__import__("os").environ.get("JEPA_ROOT", "."))
L = pickle.load(open(HERE / "labels.pkl", "rb"))
V, T = L["val"], L["train"]


def window_starts(lengths, horizon, stride):
    out = []
    for i, n in enumerate(lengths):
        tr = int(n) - 1
        out.extend((i, s) for s in range(0, tr - horizon + 1, stride))
    return np.asarray(out, np.int64).reshape(-1, 2)


def med(x):
    return float(np.median(np.asarray(x, np.float64))) if len(x) else None


def auroc(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, bool)
    p, n = labels.sum(), (~labels).sum()
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - p * (p + 1) / 2) / (p * n))


tg = V["targets"]
W = window_starts(V["lengths"], 16, 4)
starts = V["offsets"][W[:, 0]] + W[:, 1]
h = 8
tgt = starts + h
dropped = tg["apple_dropped"][:, 0] > 0.5
valid = ~dropped[starts] & ~dropped[tgt]
pma = tg["palm_minus_apple"]
disp = np.linalg.norm(pma[tgt] - pma[starts], axis=1)
moving = valid & (disp >= 0.01)
out = {
    "windows": len(W),
    "valid": int(valid.sum()),
    "moving": int(moving.sum()),
    "true_disp_median_moving": med(disp[moving]),
}
npz = np.load(MAIN / "outputs/task052-decomposition-final/leworldmodel_onboard-errors.npz")
out["npz_moving_mask_equal"] = bool(np.array_equal(npz["moving"], moving))

# ---- palm-apple: oracle persistence (true offset at start) and blind train-mean prior
tr_mean_pma = T["targets"]["palm_minus_apple"].mean(0)
out["pma_all_valid_true_persistence_median"] = med(disp[valid])
out["pma_all_valid_fraction_static_lt_1cm"] = float((disp[valid] < 0.01).mean())
out["pma_all_valid_train_mean_prior_median"] = med(np.linalg.norm(pma[tgt] - tr_mean_pma, axis=1)[valid])
out["pma_moving_train_mean_prior_median"] = med(np.linalg.norm(pma[tgt] - tr_mean_pma, axis=1)[moving])

# ---- G3 apple-plate, valid windows, full 3-D vector
amp = tg["apple_minus_plate"]
out["G3_true_persistence_median"] = med(np.linalg.norm(amp[tgt] - amp[starts], axis=1)[valid])
tr_mean_amp = T["targets"]["apple_minus_plate"].mean(0)
tr_med_amp = np.median(T["targets"]["apple_minus_plate"], axis=0)
out["G3_train_mean_prior_median"] = med(np.linalg.norm(amp[tgt] - tr_mean_amp, axis=1)[valid])
out["G3_train_median_prior_median"] = med(np.linalg.norm(amp[tgt] - tr_med_amp, axis=1)[valid])

# ---- G4 apple height, grasp cohort (phase CLOSE/LIFT at target)
phase = V["phase"][tgt]
grasp = valid & np.isin(phase, (2, 3))
lift = valid & (phase == 3)
ht = tg["apple_height"][:, 0]
out["G4_grasp_windows"] = int(grasp.sum())
out["G4_const_zero_median_abs"] = med(np.abs(ht[tgt])[grasp])
out["G4_train_mean_prior_median_abs"] = med(np.abs(ht[tgt] - T["targets"]["apple_height"][:, 0].mean())[grasp])
out["G4_true_persistence_median_abs"] = med(np.abs(ht[tgt] - ht[starts])[grasp])
out["G4_grasp_cohort_fraction_close_phase"] = float((phase[grasp] == 2).mean())

# ---- G5 apple_held AUROC, lift cohort
held = tg["apple_held"][:, 0] > 0.5
out["G5_pos"] = int(held[tgt][lift].sum())
out["G5_neg"] = int((~held[tgt])[lift].sum())
out["G5_persistence_held_at_start_auroc"] = auroc(held[starts][lift].astype(float), held[tgt][lift])
out["G5_persistence_contact_at_start_auroc"] = auroc(tg["hand_contact"][starts, 0][lift], held[tgt][lift])
out["G5_persistence_height_at_start_auroc"] = auroc(ht[starts][lift], held[tgt][lift])

# ---- decomposition: difference of medians vs per-window, and floor check
arms = {"A": "leworldmodel", "B": "leworldmodel_onboard", "C": "leworldmodel_v2_readout", "D": "leworldmodel_fine"}
E = {a: np.load(MAIN / f"outputs/task052-decomposition-final/{s}-errors.npz") for a, s in arms.items()}
for a, e in E.items():
    m = e["moving"]
    r, et, es = e["rollout"][m].astype(np.float64), e["encoded_target"][m].astype(np.float64), e["encoded_start"][m].astype(np.float64)
    out[f"{a}_rollout_med"] = med(r)
    out[f"{a}_enc_target_med"] = med(et)
    out[f"{a}_median_of_per_window_excess"] = med(r - et)
    out[f"{a}_frac_rollout_below_encoded_target"] = float((r < et).mean())
    out[f"{a}_encoder_share_ratio_of_medians"] = med(et) / med(r)

# ---- episode-clustered paired bootstrap of the three contrasts
ep = W[:, 0]
mv = E["A"]["moving"]
eps = np.unique(ep[mv])
out["moving_episodes"] = len(eps)
rng = np.random.default_rng(20520)
idx_by_ep = {e: np.flatnonzero(mv & (ep == e)) for e in eps}
B = 5000
draws = rng.integers(0, len(eps), size=(B, len(eps)))
contr = (("A_minus_B", "B", "A"), ("C_minus_A", "A", "C"), ("D_minus_B", "B", "D"))
for field in ("rollout", "encoded_target"):
    res = {k: [] for k, _, _ in contr}
    for d in draws:
        idx = np.concatenate([idx_by_ep[eps[j]] for j in d])
        for k, base, ch in contr:
            res[k].append(np.median(E[ch][field][idx].astype(np.float64)) - np.median(E[base][field][idx].astype(np.float64)))
    for k, base, ch in contr:
        lo, hi = np.percentile(res[k], [2.5, 97.5])
        pt = med(E[ch][field][mv]) - med(E[base][field][mv])
        out[f"cluster_boot_{k}_{field}"] = [round(pt * 100, 3), round(lo * 100, 3), round(hi * 100, 3)]
print(json.dumps(out, indent=1))
