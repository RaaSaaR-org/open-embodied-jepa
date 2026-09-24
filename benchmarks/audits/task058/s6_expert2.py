"""Labels-only reproduction of the BC train target distribution (no image decode)."""
import sys

import numpy as np

REPO = __import__("os").environ.get("JEPA_ROOT", ".")
sys.path.insert(0, REPO + "/src")
from embodied_jepa import readout_labels  # noqa: E402
from embodied_jepa.cloning import DISPLACEMENT_LIMIT_M, is_surviving_root  # noqa: E402
from embodied_jepa.data import DatasetStore  # noqa: E402

def extra(A):
    names=("dx","dy","dz","droll","dpitch","dyaw","grasp"); mx=(0.4,0.4,0.4,0.5,0.5,0.5,1.0)
    for i,n in enumerate(names):
        a=np.abs(A[:,i]); m=mx[i]
        at=(a>=m-1e-6).mean(); band=((a>=0.95*m)&(a<m-1e-6)).mean()
        print("   %s at|max| %.4f band95 %.4f ratio %.1f max %.4f q99 %.4f q999 %.4f" % (n,at,band,at/max(band,1e-12),a.max(),np.quantile(a,.99),np.quantile(a,.999)))
    rot=np.abs(A[:,3:6]); print("   rotation any-dim at 0.5 %.4f, per-element %.4f"%((rot>=0.5-1e-6).any(1).mean(),(rot>=0.5-1e-6).mean()))
store = DatasetStore(REPO + "/data/apple-wide-v1")
rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
for split in ("train",):
    ids = [e for e in store.manifest["splits"][split] if is_surviving_root(e)]
    dz_all, phase_all, step0 = [], [], []
    for e in ids:
        lab = readout_labels.load_privileged(
            store.root, rows[e], acknowledge_privileged_training_labels=True
        )
        base = np.asarray(lab["collector__base_action"], np.float32)
        ph = np.asarray(lab["collector__phase_index"], np.int16)
        apple = np.asarray(lab["privileged__apple_position_world"], np.float32)
        dropped = np.asarray(lab["privileged__apple_dropped"], bool)
        T = len(base)
        drift = np.linalg.norm(apple - apple[0], axis=1)[:T]
        displaced = (drift > DISPLACEMENT_LIMIT_M) & (ph[:T] < 2)
        keep = ~dropped[:T] & ~displaced
        dz_all.append(base[keep][:, [6, 7, 8, 9, 10, 11, 13]])
        phase_all.append(ph[:T][keep])
        step0.append(base[0, [6, 7, 8, 9, 10, 11, 13]])
    A = np.concatenate(dz_all)
    P = np.concatenate(phase_all)
    dz = A[:, 2]; extra(A)
    print(split, "roots", len(ids), "rows", len(A))
    print("  |dz|>=0.4 %.5f  +0.4 %.5f  -0.4 %.5f  mean %.5f  median|dz| %.4f  median dz %.4f"
          % ((np.abs(dz) >= 0.4 - 1e-6).mean(), (dz >= 0.4 - 1e-6).mean(),
             (dz <= -0.4 + 1e-6).mean(), dz.mean(), np.median(np.abs(dz)), np.median(dz)))
    for p in sorted(set(P.tolist())):
        m = P == p
        d = dz[m]
        print("   phase %d rows %6d |dz|=0.4 %.3f +0.4 %.3f -0.4 %.3f median|dz| %.4f mean dz %+.4f"
              " median|dx| %.4f" % (p, m.sum(), (np.abs(d) >= 0.4 - 1e-6).mean(),
                                    (d >= 0.4 - 1e-6).mean(), (d <= -0.4 + 1e-6).mean(),
                                    np.median(np.abs(d)), d.mean(), np.median(np.abs(A[m, 0]))))
    S = np.array(step0)
    print("  step0 dx", np.round(np.quantile(S[:, 0], [0, .1, .5, .9, 1]), 3),
          "dz", np.round(np.quantile(S[:, 2], [0, .5, 1]), 3), "grasp", np.unique(S[:, 6]))
    print("  step0 dx==+0.4 frac %.3f, in open interval frac %.3f" % (
        (S[:, 0] >= 0.4 - 1e-6).mean(), (np.abs(S[:, 0]) < 0.4 - 1e-6).mean()))
