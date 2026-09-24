import json

import numpy as np

exec(open(__file__.replace("s2_baselines3.py", "s2_baselines.py")).read().split("tr, _, _ = split")[0])
tr, offs, lens = split("train")
print("train obs", len(tr["pma"]), "h16 windows", int(sum(max(0, (n - 1) - 16 + 1) for n in lens)))
print("dropped frames", int(tr["dropped"].sum()), float(tr["dropped"].mean()))
eps = sum(1 for o, n in zip(offs, lens) if tr["dropped"][o : o + n].any())
print("episodes with dropped", eps)
C = __import__("os").environ.get("JEPA_ROOT", ".") + "/checkpoints/task050-wm-v2/"
for b in ("leworldmodel", "leworldmodel_hand_crop", "native_jepa"):
    vals = []
    for line in open(C + b + ".metrics.jsonl"):
        r = json.loads(line)
        if "selection" in r or "validation" in r:
            v = r.get("selection") or r.get("validation")
            vals.append((r.get("step"), v.get("score"), v.get("eligible")))
    print(b, vals[:20])
    rj = json.load(open(C + b + ".run.json"))
    print(" run keys", list(rj.keys())[:30])
