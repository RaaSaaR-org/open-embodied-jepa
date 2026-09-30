"""TASK-074 memfix diagnosis: stage_train's code path, step for step, on the smoke far corpus
cycled to the real stage's 270 far roots plus the real apple-look-v2-linux evidence corpus
(190 roots). A labelled PSS timeline is written every 0.25 s. Nothing here reads the real
apple-far-shift-v2 corpus."""

import os
import sys
import threading
import time
import json
from pathlib import Path

WT = Path(os.environ.get("WT", "/home/huhn/develop/emai/worktrees/task074-memfix"))
sys.argv = [sys.argv[0]] + sys.argv[1:]
import importlib.util

spec = importlib.util.spec_from_file_location("runner", WT / "scripts/run_lewm_planner_v2.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
import numpy as np

from embodied_jepa import run_guards as rg

OUT = Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=False)
UPDATES = int(os.environ.get("UPDATES", "6000"))
PHASE = ["start"]
TIMELINE = []
GIB = 2**30


def sampler():
    t0 = time.monotonic()
    while True:
        m = rg.process_tree_memory()
        TIMELINE.append((round(time.monotonic() - t0, 2), PHASE[0], m["pss"] / GIB))
        time.sleep(0.25)


threading.Thread(target=sampler, daemon=True).start()


def phase(name):
    time.sleep(0.3)
    PHASE[0] = name
    m = rg.process_tree_memory()
    print(f"[{time.strftime('%H:%M:%S')}] {name}: pss {m['pss'] / GIB:.2f} GiB", flush=True)


from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_offline as off
from embodied_jepa import pretrained_encoder as pe
from embodied_jepa import first_policy_v2_m2 as fm


class Cycled:
    """A far reader whose manifest lists the real stage's split sizes, each id served by a smoke
    episode (decoded afresh every time, as the real reader does)."""

    def __init__(self, reader, counts):
        self.r = reader
        pool = [e for s in ("train", "val") for e in reader.manifest["splits"][s]]
        self.map, splits = {}, {}
        k = 0
        for split, n in counts.items():
            splits[split] = []
            for j in range(n):
                eid = f"cyc-{split}-{j}"
                self.map[eid] = pool[k % len(pool)]
                k += 1
                splits[split].append(eid)
        self.manifest = {"splits": splits}

    def episode(self, eid, keys):
        return self.r.episode(self.map[eid], keys=keys)


phase("load")
smoke = rt2.CorpusReader(
    Path("/home/huhn/develop/emai/worktrees/task074-prereg/outputs/task074-scratch/frozen2/far-corpus"),
    "68499d90c4ee4e0c4cf76b861ed1eafc657df1bcee26f6913d803c4015d11c46",
    splits=("train", "val"),
)
reader = Cycled(smoke, {"train": 240, "val": 30})
evidence = Path("/home/huhn/develop/emai/worktrees/task072-run")
look = rt2.CorpusReader(
    evidence / fm.EVIDENCE["corpus"], fm.EVIDENCE["corpus_manifest_sha256"], splits=("train", "val")
)
encoder = pe.load_pretrained()
band = lp.FEATURE_BAND
band_len = band[1] - band[0] + 1
sources = [("far", reader, s) for s in lp.READ_SPLITS] + [("look", look, s) for s in ("train", "val")]
n_roots = sum(len(rd.manifest["splits"][sp]) for _k, rd, sp in sources)
table = off.Table(capacity=n_roots * band_len)
plate_rows = {"train": [], "val": []}
keys = (*rt2.EPISODE_ARRAYS, *lp.EXTRA_EPISODE_ARRAYS)
phase("features")
for source, rd, split in sources:
    for episode_id in rd.manifest["splits"][split]:
        arrays, meta = rd.episode(episode_id, keys=keys if source == "far" else rt2.EPISODE_ARRAYS)
        frames = arrays["frames"]
        hi = min(band[1], len(frames) - 1)
        if hi <= band[0] + lp.CHUNK:
            continue
        sl = slice(band[0], hi + 1)
        if source == "far":
            plate = arrays["plate"][sl]
        else:
            plate = np.repeat(np.asarray([[*meta["truth_xy"][2:], 0.746]]), hi + 1 - band[0], 0)
        pooled = off.pooled_features(encoder, frames[sl], device="cuda")
        root = {
            "id": episode_id,
            "split": split if source == "far" else f"look-{split}",
            "source": source,
            "shift_step": (meta.get("shift") or {}).get("step") if source == "far" else None,
        }
        M.band_root(table, root, arrays, pooled, plate)
        if source == "far":
            for t in lp.DECISION_STEPS:
                if t < len(frames):
                    plate_rows[split].append((episode_id, frames[t], arrays["plate"][t, :2]))
table.seal()
phase("sealed")
print("frames", table.size, "roots", len(table.roots), flush=True)
tok_train = off.full_tokens_cpu(encoder, np.stack([r[1] for r in plate_rows["train"]]))
phase("r_plate_fit")
r_plate = off.fit_r_plate(
    tok_train, np.stack([r[2] for r in plate_rows["train"]]), [r[0] for r in plate_rows["train"]]
)
del tok_train
phase("r_off_fit")
idx = {s: [i for i, r in enumerate(table.roots) if r["split"] == s] for s in ("train", "val", "look-train", "look-val")}
r_off, _rows = off.fit_r_off(table, idx["train"])
phase("val_tokens")
tok_val = off.full_tokens_cpu(encoder, np.stack([r[1] for r in plate_rows["val"]]))
del tok_val
phase("baselines")
o2_starts, o2_owners = table.windows(idx["val"], start_steps=set(lp.DECISION_STEPS))
off.o2_statistics({}, {}, r_off, table, o2_starts, o2_owners)
phase("train_context")
schema = rt2.make_robot().state_schema
ctx = M.train_context(table, idx, schema, {"diag": True})
print("train windows", len(ctx["train_starts"]), "val windows", len(ctx["val_starts"]), flush=True)
phase("calibration")
t = time.monotonic()
_m, rec = off.train_model(ctx, "W", lp.MODEL_SEEDS[0], UPDATES, 1000)
print("calibration seconds", time.monotonic() - t, "curve", rec["val_curve"], flush=True)
phase("end")
time.sleep(1)
by = {}
for _t, p, v in TIMELINE:
    by[p] = max(by.get(p, 0), v)
print(json.dumps(by, indent=1))
(OUT / "timeline.json").write_text(json.dumps({"timeline": TIMELINE, "phase_peak_gib": by, "val_curve": rec["val_curve"]}))
