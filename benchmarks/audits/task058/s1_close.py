import json, glob, numpy as np, os
base=__import__("os").environ.get("JEPA_ROOT", ".") + "/outputs/"
for run in ["apple-wide-grasp-closure-v3","apple-wide-object-ceiling-v2"]:
    print("==",run)
    for d in sorted(glob.glob(base+run+"/attempts/*-privileged_object")):
        rows=[json.loads(l) for l in open(d+"/trace.jsonl")]
        res=[r for r in rows if r.get("event")=="result" and r.get("phase")=="close"]
        if not res:
            print(os.path.basename(d),"no close results"); continue
        z=np.array([r["applied_action"][8] for r in res])
        lat=np.array([r["applied_action"][6:8]+r["applied_action"][9:12] for r in res])
        g=np.array([r["applied_action"][13] for r in res])
        a0=np.array(res[0]["live_apple"])
        disp=max(np.linalg.norm(np.array(r["live_apple"][:2])-a0[:2]) for r in res)
        print(os.path.basename(d).split("-")[0], len(res), "z mean %.3f min %.3f max %.3f std %.3f"%(z.mean(),z.min(),z.max(),z.std()), "n_at_-0.5",int((np.isclose(z,-0.5)).sum()), "lat|max| %.3f"%np.abs(lat).max(), "grasp uniq",np.unique(np.round(g,2))[:5], "maxdisp_cm %.2f"%(100*disp))
