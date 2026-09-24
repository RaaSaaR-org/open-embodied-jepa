import json, numpy as np
base=__import__("os").environ.get("JEPA_ROOT", ".") + "/outputs/apple-wide-object-ceiling-v2/attempts/"
for s in [45100,45103,45105,45000]:
    rows=[json.loads(l) for l in open(base+f"{s}-privileged_object/trace.jsonl")]
    res=[r for r in rows if r.get("event")=="result" and r.get("phase")=="close"]
    a0=np.array(res[0]["live_apple"])
    d=[100*np.linalg.norm(np.array(r["live_apple"][:2])-a0[:2]) for r in res]
    z=[r["live_apple"][2] for r in res]
    print(s, [round(r["live_apple"][2],3) for r in res][::4], " ".join("%.1f"%x for x in d), "| final z %.3f"%z[-1], "argmax",int(np.argmax(d)))
    # last row overall
    last=[r for r in rows if "live_apple" in r][-1]
    print("   last live apple", np.round(last["live_apple"],3), last.get("phase"))
