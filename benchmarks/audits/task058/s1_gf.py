import json, numpy as np
root=__import__("os").environ.get("JEPA_ROOT", ".") + "/data/apple-wide-v1"
m=json.load(open(root+"/meta/jepa_manifest.json"))
from collections import Counter
c=Counter(); lens=[]
for e in m["episodes"]:
    o=e["metadata"]["privileged_outcome_labels"]
    if o["grasp_phase_failure"]:
        c[(e["metadata"]["kind"], e["metadata"]["termination"])]+=1
        if e["metadata"]["termination"]=="guard_refused": lens.append((e["metadata"]["kind"], e["length"]-1 - (e["metadata"].get("pre_grasp_frame", e["metadata"].get("branch_frame_index",0)) if e["metadata"]["kind"]=="root" else 0)))
print(c)
b=[l for k,l in lens if k=="branch"]; r=[l for k,l in lens if k=="root"]
print("branch guard fail post-branch cmds: n",len(b),"min/med/max",min(b),np.median(b),max(b), "lt 45 (close not finished)", sum(x<45 for x in b))
print("root guard fail cmds after pregrasp:", sorted(r))
# branch_complete failures by kind
k=Counter((e["metadata"].get("branch_kind"),e["metadata"]["termination"]) for e in m["episodes"] if e["metadata"]["kind"]=="branch" and e["metadata"]["privileged_outcome_labels"]["grasp_phase_failure"])
print(k)
