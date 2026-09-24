"""S5 audit: blind baselines for P1-P3 and for the BC offline selection score. Read-only."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WT = __import__("os").environ.get("JEPA_ROOT", ".")
sys.path.insert(0, WT + "/src")
from embodied_jepa import readout_labels  # noqa: E402

ROOT = Path(__import__("os").environ.get("JEPA_ROOT", ".") + "/data/apple-wide-v1")
man = json.load(open(ROOT / "meta/jepa_manifest.json"))
rows = {r["episode_id"]: r for r in man["episodes"]}
FROZEN = tuple(range(48000, 48200))
AIM = {s for i, s in enumerate(FROZEN) if i % 5 == 4}


def surviving(e):
    p = e.split("-")
    return len(p) == 2 and int(p[1]) in FROZEN and int(p[1]) not in AIM


# proprio states by episode_index
states = {}
for f in sorted(glob.glob(str(ROOT / "data/*/*.parquet"))):
    t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
    ei = t.column("episode_index").to_numpy()
    fi = t.column("frame_index").to_numpy()
    st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
    for e in np.unique(ei):
        m = ei == e
        o = np.argsort(fi[m])
        states[int(e)] = st[m][o]


def load(split):
    out = {k: [] for k in ("pma", "apos", "drop", "phase", "base", "has", "disp", "state", "ep")}
    for e in man["splits"][split]:
        if not surviving(e):
            continue
        row = rows[e]
        lab = readout_labels.load_privileged(ROOT, row, acknowledge_privileged_training_labels=True)
        rest = float(lab["privileged__apple_position_world"][0, 2])
        tg = readout_labels.targets(lab, rest)
        n = len(tg["apple_position"])
        ph = np.asarray(lab["collector__phase_index"], np.int16)
        phase = np.full(n, -1, np.int16)
        phase[: len(ph)] = ph
        phase[len(ph):] = ph[-1]
        base = np.zeros((n, 14), np.float32)
        b = np.asarray(lab["collector__base_action"], np.float32)
        base[: len(b)] = b
        has = np.zeros(n, bool)
        has[: len(b)] = True
        drift = np.linalg.norm(tg["apple_position"] - tg["apple_position"][0], axis=1)
        disp = (drift > 0.01) & (phase < 2)
        st = states[row["episode_index"]]
        assert len(st) == n, (e, len(st), n)
        out["pma"].append(tg["palm_minus_apple"])
        out["apos"].append(tg["apple_position"])
        out["drop"].append(tg["apple_dropped"][:, 0] > 0.5)
        out["phase"].append(phase)
        out["base"].append(base)
        out["has"].append(has)
        out["disp"].append(disp)
        out["state"].append(st)
        out["ep"].append(np.full(n, int(e.split("-")[1])))
    return {k: np.concatenate(v) for k, v in out.items()}


tr, va = load("train"), load("val")
print("surviving rows train/val", len(tr["phase"]), len(va["phase"]))
vk = ~va["drop"]
tk = ~tr["drop"]
print("val rows scored", vk.sum(), "dropped", va["drop"].sum())
res = {}
cohorts = {"P1_close": [2], "P2_approach": [0, 1], "P3_orient": [0]}
for name, phs in cohorts.items():
    tgt = "apos" if name == "P3_orient" else "pma"
    vm = vk & np.isin(va["phase"], phs)
    # (a) per-phase train mean / median (prior-only; phase is known)
    pred = np.zeros_like(va[tgt][vm])
    predm = np.zeros_like(va[tgt][vm])
    for p in phs:
        tm = tk & (tr["phase"] == p)
        sel = va["phase"][vm] == p
        pred[sel] = tr[tgt][tm].mean(0)
        predm[sel] = np.median(tr[tgt][tm], 0)
    ea = np.linalg.norm(pred - va[tgt][vm], axis=1)
    em = np.linalg.norm(predm - va[tgt][vm], axis=1)
    # (b) per-phase linear regression on proprio state (no vision), fitted on train
    X = lambda d, m: np.c_[d["state"][m], np.ones(m.sum())]  # noqa: E731
    predr = np.zeros_like(va[tgt][vm])
    for p in phs:
        tm = tk & (tr["phase"] == p)
        W, *_ = np.linalg.lstsq(X(tr, tm), tr[tgt][tm], rcond=None)
        sel = np.zeros(len(va["phase"]), bool)
        sel[vm] = va["phase"][vm] == p
        predr[va["phase"][vm] == p] = X(va, sel) @ W
    er = np.linalg.norm(predr - va[tgt][vm], axis=1)
    # pooled-over-phases single linear regression too
    tm = tk & np.isin(tr["phase"], phs)
    W, *_ = np.linalg.lstsq(X(tr, tm), tr[tgt][tm], rcond=None)
    ep = np.linalg.norm(X(va, vm) @ W - va[tgt][vm], axis=1)
    res[name] = dict(
        rows=int(vm.sum()),
        per_phase_train_mean_cm=100 * float(np.median(ea)),
        per_phase_train_median_cm=100 * float(np.median(em)),
        proprio_linear_per_phase_cm=100 * float(np.median(er)),
        proprio_linear_pooled_cm=100 * float(np.median(ep)),
    )
print(json.dumps(res, indent=1))
# per-phase table baseline for all phases
for p in range(7):
    vm = vk & (va["phase"] == p)
    tm = tk & (tr["phase"] == p)
    if not vm.sum():
        continue
    e1 = np.median(np.linalg.norm(va["pma"][vm] - tr["pma"][tm].mean(0), axis=1))
    e2 = np.median(np.linalg.norm(va["apos"][vm] - tr["apos"][tm].mean(0), axis=1))
    print("phase", p, "rows", vm.sum(), "prior pma cm %.3f  apos cm %.3f" % (100 * e1, 100 * e2))

# BC offline: selection score of constant predictors
FREE = [6, 7, 8, 9, 10, 11, 13]
ttrain = tk & tr["has"] & ~tr["disp"]
tval = vk & va["has"] & ~va["disp"]
print("BC rows train/val", ttrain.sum(), tval.sum())
T = tr["base"][ttrain][:, FREE]
V = va["base"][tval][:, FREE]
names = ["right_dx", "right_dy", "right_dz", "right_droll", "right_dpitch", "right_dyaw", "right_grasp"]
print("train |dz|==0.4 rate %.4f  dz==+0.4 %.4f dz==-0.4 %.4f" % (
    np.mean(np.isclose(np.abs(T[:, 2]), 0.4)), np.mean(np.isclose(T[:, 2], 0.4)),
    np.mean(np.isclose(T[:, 2], -0.4))))
for label, c in (("zero", np.zeros(7)), ("train_median", np.median(T, 0)), ("train_mean", T.mean(0))):
    err = np.abs(V - c)
    print(label, "pooled median %.5f" % np.median(err),
          "per-dim", {n: round(float(np.median(err[:, i])), 4) for i, n in enumerate(names)})
print("fraction of val target entries exactly equal to train per-dim median:",
      float(np.mean(V == np.median(T, 0))))
# per-phase constant (prior-only phase-conditioned) predictor
pv = va["phase"][tval]
pt = tr["phase"][ttrain]
pred = np.zeros_like(V)
for p in np.unique(pv):
    pred[pv == p] = np.median(T[pt == p], 0)
err = np.abs(V - pred)
print("per-phase train median pooled median %.5f" % np.median(err),
      {n: round(float(np.median(err[:, i])), 4) for i, n in enumerate(names)})
