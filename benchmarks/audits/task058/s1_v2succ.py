import json, numpy as np
base=__import__("os").environ.get("JEPA_ROOT", ".") + "/outputs/apple-wide-object-ceiling-v2/attempts/"
for s in [45001,45101,45002]:
    rows=[json.loads(l) for l in open(base+f"{s}-privileged_object/trace.jsonl")]
    res=[r for r in rows if r.get("event")=="result" and r.get("phase")=="close"]
    a0=np.array(res[0]["live_apple"])
    d=[100*np.linalg.norm(np.array(r["live_apple"][:2])-a0[:2]) for r in res]
    print(s," ".join("%.1f"%x for x in d), [r["live_contact"] for r in res][::5])
