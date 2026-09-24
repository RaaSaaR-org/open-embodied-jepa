"""Step-zero expert commands: recorded collector (apple_collector_policy) vs the command
OracleManipulationPolicy would issue at the same state, reconstructed from privileged labels.
Validity self-check: the reconstructed collector command must match the recorded base action."""
import sys

import numpy as np

REPO = __import__("os").environ.get("JEPA_ROOT", ".")
sys.path.insert(0, REPO + "/src")
from embodied_jepa import readout_labels  # noqa: E402
from embodied_jepa.cloning import is_surviving_root  # noqa: E402
from embodied_jepa.data import DatasetStore  # noqa: E402

TPS = 0.015
store = DatasetStore(REPO + "/data/apple-wide-v1")
rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
for split in ("train", "val"):
    ids = [e for e in store.manifest["splits"][split] if is_surviving_root(e)]
    rec, col, ora = [], [], []
    for e in ids:
        lab = readout_labels.load_privileged(
            store.root, rows[e], acknowledge_privileged_training_labels=True
        )
        base = np.asarray(lab["collector__base_action"], np.float32)[0, 6:9]
        pma = np.asarray(lab["privileged__palm_minus_apple_world"], np.float32)[0]
        # orient target offset from apple: oracle [-0.03, 0, 0.13]; collector adds +0.015 in x
        o = np.clip((np.array([-0.03, 0, 0.13]) - pma) / TPS, -0.4, 0.4)
        c = np.clip((np.array([-0.03 + 0.015, 0, 0.13]) - pma) / TPS, -0.4, 0.4)
        rec.append(base); col.append(c); ora.append(o)
    rec, col, ora = map(np.array, (rec, col, ora))
    print(split, len(ids), "reconstruction max|err| vs recorded (dx,dy,dz):",
          np.round(np.abs(rec - col).max(0), 4), "median", np.round(np.median(np.abs(rec - col), 0), 4))
    d = np.abs(rec - ora)
    print("  |recorded collector - oracle| step0 dx: median %.4f, frac>0.057 %.3f, max %.3f"
          % (np.median(d[:, 0]), (d[:, 0] > 0.057).mean(), d[:, 0].max()),
          " dy/dz max", np.round(d[:, 1:].max(0), 4))
