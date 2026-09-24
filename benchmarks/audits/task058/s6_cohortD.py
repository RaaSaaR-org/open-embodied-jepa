import json
import sys
from math import comb

import numpy as np

REPO = __import__("os").environ.get("JEPA_ROOT", ".")
sys.path.insert(0, REPO + "/src")
from embodied_jepa import readout_labels  # noqa: E402
from embodied_jepa.cloning import is_surviving_root  # noqa: E402
from embodied_jepa.data import DatasetStore  # noqa: E402

store = DatasetStore(REPO + "/data/apple-wide-v1")
rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
palms = []
for e in [e for e in store.manifest["splits"]["train"] if is_surviving_root(e)]:
    lab = readout_labels.load_privileged(
        store.root, rows[e], acknowledge_privileged_training_labels=True
    )
    palms.append(
        np.asarray(lab["privileged__palm_minus_apple_world"])[0]
        + np.asarray(lab["privileged__apple_position_world"])[0]
    )
palms = np.array(palms)
print("home palm world spread (std xyz):", np.round(palms.std(0), 5), "mean", np.round(palms.mean(0), 4))
home = palms.mean(0)
r = json.load(open(REPO + "/outputs/task056-cohort-d/a2.json"))
print("reset keys", list(next(iter(r["resets"].values())).keys()))
diffs = []
for seed, v in r["resets"].items():
    ax = v["object_xy"][0]
    pma_x = home[0] - ax
    o = np.clip((-0.03 - pma_x) / 0.015, -0.4, 0.4)
    c = np.clip((-0.015 - pma_x) / 0.015, -0.4, 0.4)
    diffs.append(abs(c - o))
    print(seed, "collector dx %.3f oracle dx %.3f diff %.3f" % (c, o, abs(c - o)))
d = np.array(diffs)
print("median over 16 = %.4f ; seeds with diff > 0.057: %d/16" % (np.median(d), (d > 0.057).sum()))
p = 0.38
print("P(X>=8|16,0.38)=%.3f" % sum(comb(16, k) * p**k * (1 - p) ** (16 - k) for k in range(8, 17)))
