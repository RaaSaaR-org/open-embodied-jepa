import json, numpy as np
base=__import__("os").environ.get("JEPA_ROOT", ".") + "/outputs/apple-wide-object-ceiling-v2/attempts/"
for s in [45103,45100,45105,45000]:
    rows=[json.loads(l) for l in open(base+f"{s}-privileged_object/trace.jsonl")]
    res=[r for r in rows if r.get("event")=="result"]
    ci=[i for i,r in enumerate(res) if r["phase"]=="close"]
    a0=np.array(res[ci[0]]["live_apple"])
    print(s)
    for i,r in enumerate(res):
        if i>=ci[0]-1 and (i-ci[0])%3==0 or i==len(res)-1:
            sc=r["score"]
            print("  ",i-ci[0],r["phase"],"contact",r["live_contact"],"apple",np.round(np.array(r["live_apple"])-a0,3),"palm-apple",np.round(np.array(r["live_palm"])-np.array(r["live_apple"]),3),"sc_contact",sc.get("hand_contact"),"h",round(sc["object_height_m"],3))
    rep=json.load(open(base+f"{s}-privileged_object/report.json"))
    print("  term",rep.get("termination_reason"), rep.get("phase"))
