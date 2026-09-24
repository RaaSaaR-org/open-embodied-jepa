"""Follow-ups: G6 constant details, G4 cohort composition, dropped persistence AUROC, train all-window persistence."""
import sys

import numpy as np

sys.argv = ["x"]
exec(open(__file__.replace("s2_baselines2.py", "s2_baselines.py")).read().split("tr, _, _ = split")[0])

tr, troffs, trlens = split("train")
va, offs, lens = split("val")


def win(offs, lens):
    s = []
    for o, n in zip(offs, lens):
        s += [o + k for k in range(0, (n - 1) - 16 + 1, 4)]
    return np.array(s)


ok = ~tr["dropped"]
trct = np.linalg.norm(tr["pma"][ok] - APPROACH, axis=1)
trapp = trct[np.isin(tr["phase"][ok], (0, 1)) & (trct >= 0.02)]
c0 = float(np.median(trapp))
starts = win(offs, lens)
H = 8
t = starts + H
d = va["dropped"]
valid = ~d[starts] & ~d[t]
ph = va["phase"][t]
appr = valid & np.isin(ph, (0, 1))
ct = np.linalg.norm(va["pma"][t] - APPROACH, axis=1)
rows_ = appr & (ct >= 0.02)
print("c0 cm", 100 * c0, "val approach cost quantiles cm", np.round(100 * np.quantile(ct[rows_], [0, .1, .25, .5, .75, .9, 1]), 2))
print("phase split of approach rows (orient, descend):", int((ph[rows_] == 0).sum()), int((ph[rows_] == 1).sum()))
# val-median constant too (leaky, for reference)
print("G6 val-median const", float(np.median(np.abs(np.log(np.median(ct[rows_]) / ct[rows_])))))
grasp = valid & np.isin(ph, (2, 3))
hh = va["h"][t][grasp]
print("grasp cohort: frac |h+3.4mm|<1cm", float((np.abs(hh + 0.0034) < 0.01).mean()), "frac h>=2cm", float((hh >= 0.02).mean()))
print("grasp cohort quantiles of h (cm)", np.round(100 * np.quantile(hh, [.1, .25, .5, .6, .7, .75, .9]), 3))
lift = valid & (ph == 3)
print("lift-cohort frac h>=2cm", float((va["h"][t][lift] >= 0.02).mean()))
# dropped AUROC persistence (all windows, as code does)
print("dropped persistence AUROC h8 all windows", auroc(d[starts].astype(float), d[t]), "pos", int(d[t].sum()), "of", len(t))
# train persistence on all windows (protocol line 238 says 0.9 cm)
ts = win(troffs, trlens)
tt = ts + 8
disp = np.linalg.norm(tr["pma"][tt] - tr["pma"][ts], axis=1)
tv = ~tr["dropped"][ts] & ~tr["dropped"][tt]
print("train truth persistence h8: all windows", 100 * np.median(disp), "valid", 100 * np.median(disp[tv]))
ts1 = []
for o, n in zip(troffs, trlens):
    ts1 += [o + k for k in range(0, (n - 1) - 8 + 1)]
ts1 = np.array(ts1)
disp1 = np.linalg.norm(tr["pma"][ts1 + 8] - tr["pma"][ts1], axis=1)
print("train truth persistence h8 stride1 horizon8 windows, all", 100 * np.median(disp1))
# val all-valid: split of model-independent composition
disp8 = np.linalg.norm(va["pma"][t] - va["pma"][starts], axis=1)
moving = valid & (disp8 >= 0.01)
print("val valid", int(valid.sum()), "moving frac", float(moving.sum() / valid.sum()))
