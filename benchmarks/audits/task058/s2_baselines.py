"""Label-only blind baselines for TASK-050 gates (val; reimplements 3b6af0b window_metrics cohorts)."""
import json
from pathlib import Path

import numpy as np

from embodied_jepa import training_labels

ROOT = Path(__import__("os").environ.get("JEPA_ROOT", ".") + "/data/apple-wide-v1")
m = json.load(open(ROOT / "meta/jepa_manifest.json"))
rows = {r["episode_id"]: r for r in m["episodes"]}
APPROACH = np.array([-0.015, 0.0, 0.13], np.float32)


def labels(row):
    return training_labels.load(
        ROOT, row["metadata"]["training_labels"], groups=("privileged", "collector"),
        acknowledge_privileged_training_labels=True,
    )


def split(name):
    T = {k: [] for k in ("pma", "h", "amp", "contact", "held", "dropped", "phase")}
    offs, lens, rest, off = [], [], {}, 0
    for eid in m["splits"][name]:
        row = rows[eid]
        root = row["metadata"]["root_episode_id"]
        if root not in rest:
            rest[root] = float(labels(rows[root])["privileged__apple_position_world"][0, 2])
        L = labels(row)
        apple = np.asarray(L["privileged__apple_position_world"], np.float32)
        n = len(apple)
        h = apple[:, 2] - np.float32(rest[root])
        c = np.asarray(L["privileged__hand_contact"], bool)
        T["pma"].append(np.asarray(L["privileged__palm_minus_apple_world"], np.float32))
        T["h"].append(h)
        T["amp"].append(apple - np.asarray(L["privileged__plate_position_world"], np.float32))
        T["contact"].append(c)
        T["held"].append(c & (h >= 0.02))
        T["dropped"].append(np.asarray(L["privileged__apple_dropped"], bool))
        ph = np.asarray(L["collector__phase_index"], np.int16)
        p = np.full(n, ph[-1], np.int16)
        p[: len(ph)] = ph
        T["phase"].append(p)
        offs.append(off)
        lens.append(n)
        off += n
    return {k: np.concatenate(v) for k, v in T.items()}, np.array(offs), np.array(lens)


def auroc(s, lab):
    s = np.asarray(s, float)
    lab = np.asarray(lab, bool)
    P, N = lab.sum(), (~lab).sum()
    if not P or not N:
        return None
    o = np.argsort(s, kind="mergesort")
    rk = np.empty(len(s))
    sv = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        rk[o[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((rk[lab].sum() - P * (P + 1) / 2) / (P * N))


tr, _, _ = split("train")
ok = ~tr["dropped"]
mean_pma = tr["pma"][ok].mean(0)
med_pma = np.median(tr["pma"][ok], 0)
mean_amp = tr["amp"][ok].mean(0)
med_amp = np.median(tr["amp"][ok], 0)
med_h = float(np.median(tr["h"][ok]))
trct = np.linalg.norm(tr["pma"][ok] - APPROACH, axis=1)
trapp = trct[np.isin(tr["phase"][ok], (0, 1)) & (trct >= 0.02)]
va, offs, lens = split("val")
print("val obs", len(va["pma"]))
starts = []
for o, n in zip(offs, lens):
    starts += [o + s for s in range(0, (n - 1) - 16 + 1, 4)]
starts = np.array(starts)
print("windows", len(starts))
for H in (1, 4, 8, 16):
    t = starts + H
    d = va["dropped"]
    valid = ~d[starts] & ~d[t]
    disp = np.linalg.norm(va["pma"][t] - va["pma"][starts], axis=1)
    moving = valid & (disp >= 0.01)
    ph = va["phase"][t]
    grasp = valid & np.isin(ph, (2, 3))
    lift = valid & (ph == 3)
    appr = valid & np.isin(ph, (0, 1))
    r = {"valid": int(valid.sum()), "moving": int(moving.sum())}
    r["truth_persist_all_valid_cm"] = 100 * np.median(disp[valid])
    r["truth_persist_nonmoving_cm"] = 100 * np.median(disp[valid & ~moving])
    r["truth_persist_moving_cm"] = 100 * np.median(disp[moving])
    for nm, c in (("trainmean", mean_pma), ("trainmedian", med_pma)):
        e = np.linalg.norm(va["pma"][t] - c, axis=1)
        r[f"pma_{nm}_moving_cm"] = 100 * np.median(e[moving])
        r[f"pma_{nm}_all_valid_cm"] = 100 * np.median(e[valid])
    for nm, c in (("trainmean", mean_amp), ("trainmedian", med_amp)):
        e = np.linalg.norm(va["amp"][t] - c, axis=1)
        r[f"G3_amp_{nm}_valid_cm"] = 100 * np.median(e[valid])
    ea = np.linalg.norm(va["amp"][t] - va["amp"][starts], axis=1)
    r["G3_amp_truth_persist_valid_cm"] = 100 * np.median(ea[valid])
    r["grasp_windows"] = int(grasp.sum())
    r["grasp_close_frac"] = float((ph[grasp] == 2).mean())
    r["G4_h_trainmedian_grasp_cm"] = 100 * np.median(np.abs(va["h"][t] - med_h)[grasp])
    r["G4_h_const_-3.4mm_grasp_cm"] = 100 * np.median(np.abs(va["h"][t] + 0.0034)[grasp])
    r["G4_h_truth_persist_grasp_cm"] = 100 * np.median(np.abs(va["h"][t] - va["h"][starts])[grasp])
    ct = np.linalg.norm(va["pma"][t] - APPROACH, axis=1)
    rows_ = appr & (ct >= 0.02)
    r["G6_rows"] = int(rows_.sum())
    c0 = float(np.median(trapp))
    r["G6_const_train_approach_median"] = float(np.median(np.abs(np.log(c0 / ct[rows_]))))
    cp = np.linalg.norm(va["pma"][starts] - APPROACH, axis=1)
    lr = np.abs(np.log(np.maximum(cp, 1e-4) / np.maximum(ct, 1e-4)))
    r["G6_truth_persist"] = float(np.median(lr[rows_]))
    held = va["held"][t]
    r["G5_lift_pos"] = int(held[lift].sum())
    r["G5_lift_neg"] = int((~held[lift]).sum())
    r["G5_auroc_truth_height_at_start"] = auroc(va["h"][starts][lift], held[lift])
    r["G5_auroc_truth_held_at_start"] = auroc(va["held"][starts][lift], held[lift])
    r["G5_auroc_truth_contact_at_start"] = auroc(va["contact"][starts][lift], held[lift])
    print(H, {k: (round(float(v), 3) if v is not None and not isinstance(v, int) else v) for k, v in r.items()})
