import json, numpy as np, io
root=__import__("os").environ.get("JEPA_ROOT", ".") + "/data/apple-wide-v1"
m=json.load(open(root+"/meta/jepa_manifest.json"))
eps=m["episodes"]
drop=[e for e in eps if e["metadata"]["kind"]=="branch" and e["metadata"]["privileged_outcome_labels"]["dropped_after_grasp"]]
print(len(drop))
zs=[];kinds={}
for e in drop:
    ref=e["metadata"]["training_labels"]
    p=root+"/labels/"+e["episode_id"]+".npz"
    d=np.load(p)
    a=d["privileged__apple_position_world"]
    zs.append((e["episode_id"], float(a[-1,2]), float(a[:,2].max()), float(np.hypot(*(a[-1,:2]-a[0,:2])))))
    k=e["metadata"]["branch_kind"]; kinds[k]=kinds.get(k,0)+1
for z in zs[:10]: print(z)
print("final z range", min(z[1] for z in zs), max(z[1] for z in zs))
print(kinds)
# also roots: grasp at pre-grasp? branches whose stage grasp at row0
g0=0
for e in eps:
    if e["metadata"]["kind"]!="branch": continue
    d=np.load(root+"/labels/"+e["episode_id"]+".npz")
    if d["privileged__stages"][0,1]: g0+=1
print("branches grasped at first frame",g0)
# root grasp before branch frame?
rb=0
for e in eps:
    if e["metadata"]["kind"]!="root": continue
    bf=e["metadata"].get("branch_frame")
    d=np.load(root+"/labels/"+e["episode_id"]+".npz")
    if bf is not None and d["privileged__stages"][:bf+1,1].any(): rb+=1
print("roots grasped at/before branch frame",rb)
