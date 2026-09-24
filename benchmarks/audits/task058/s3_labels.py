"""Rebuild the val/train label arrays exactly as world_model_v2.load_split does (no images)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, __import__("os").environ.get("JEPA_ROOT", ".") + "/src")
from embodied_jepa import readout_labels  # noqa: E402

ROOT = Path(__import__("os").environ.get("JEPA_ROOT", ".") + "/data/apple-wide-v1")
OUT = Path(__file__).with_name("labels.pkl")
m = json.load(open(ROOT / "meta/jepa_manifest.json"))
rows = {r["episode_id"]: r for r in m["episodes"]}


def build(split):
    ids = list(m["splits"][split])
    rest = {}
    tg, ph, lengths = {}, [], []
    for name in ids:
        row = rows[name]
        root = row["metadata"]["root_episode_id"]
        if root not in rest:
            rl = readout_labels.load_privileged(
                ROOT, rows[root], acknowledge_privileged_training_labels=True
            )
            rest[root] = float(rl["privileged__apple_position_world"][0, 2])
        lab = readout_labels.load_privileged(ROOT, row, acknowledge_privileged_training_labels=True)
        t = readout_labels.targets(lab, rest[root])
        count = len(t["palm_minus_apple"])
        lengths.append(count)
        for k, v in t.items():
            tg.setdefault(k, []).append(v)
        p = np.asarray(lab["collector__phase_index"], np.int16)
        full = np.empty(count, np.int16)
        full[: len(p)] = p
        full[len(p) :] = p[-1]
        ph.append(full)
    lengths = np.array(lengths)
    return dict(
        ids=ids,
        lengths=lengths,
        offsets=np.concatenate(([0], np.cumsum(lengths)[:-1])),
        targets={k: np.concatenate(v) for k, v in tg.items()},
        phase=np.concatenate(ph),
        rows={i: rows[i] for i in ids},
    )


res = {s: build(s) for s in ("val", "train")}
pickle.dump(res, open(OUT, "wb"))
print({s: (len(r["ids"]), int(r["lengths"].sum())) for s, r in res.items()})
