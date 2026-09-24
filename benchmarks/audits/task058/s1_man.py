import json,sys
for f in ["apple-wide-object-ceiling-v2","apple-wide-grasp-closure-v3"]:
    d=json.load(open(f"benchmarks/manifests/{f}.json"))
    print("==",f,list(d.keys()))
    for k in ("result","run","attempts_detail"):
        if k in d: print(k, json.dumps(d[k])[:1800])
