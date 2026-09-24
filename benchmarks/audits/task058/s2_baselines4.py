import numpy as np

exec(open(__file__.replace("s2_baselines4.py", "s2_baselines.py")).read().split("tr, _, _ = split")[0])
n_any = n_last = 0
roots = set()
dropped_flag = 0
for eid in m["splits"]["train"]:
    row = rows[eid]
    L = labels(row)
    d = np.asarray(L["privileged__apple_dropped"], bool)
    n_any += bool(d.any())
    n_last += bool(d[-1])
    if d.any():
        roots.add(row["metadata"]["root_episode_id"])
    po = row["metadata"].get("privileged_outcome_labels", {})
    if po.get("dropped") or po.get("apple_dropped"):
        dropped_flag += 1
print("any", n_any, "last", n_last, "roots", len(roots), "outcome-flag", dropped_flag)
print(rows[m["splits"]["train"][0]]["metadata"]["privileged_outcome_labels"])
