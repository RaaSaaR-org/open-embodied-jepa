import json
import numpy as np

D = __import__("os").environ.get("JEPA_ROOT", ".") + "/outputs/task056-cohort-d/"
neg = 0
nvals = 0


def walk(x):
    global neg, nvals
    if isinstance(x, dict):
        for v in x.values():
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        nvals += 1
        if x < 0:
            neg += 1


for a in "a0 a1 a2 a3".split():
    r = json.load(open(D + a + ".json"))
    src = r["source"]
    print(a, r["arm"], {k: src.get(k) for k in ("revision", "dirty")}, r["result"])
    print("  keys", sorted(r.keys()))
    occ = {}
    q50s = []
    q90s = []
    q99s = []
    mx = []
    ood_n = 0
    ncmd = 0
    droll = 0
    for t in r["attempts"]:
        walk(t["command_statistics"])
        s = t["score"]
        cs = t["command_statistics"]
        bs = cs["by_stage"]
        for k, v in bs.items():
            occ[k] = occ.get(k, 0) + v["commands"]
        dz = cs["per_dimension"]["dz"]
        q50s.append(dz["q50"]); q90s.append(dz["q90"]); q99s.append(dz["q99"]); mx.append(dz["max_abs"])
        ood_n += dz["out_of_distribution_rate"] * cs["commands"]
        droll += cs["per_dimension"]["droll"]["out_of_distribution_rate"] * cs["commands"]
        ncmd += cs["commands"]
        print("  ", t["seed"], t["termination_reason"], t["executed_steps"], t["clipped_commands"],
              s, {k: v["commands"] for k, v in bs.items()},
              round(t["median_control_seconds"] * 1000, 2) if t["median_control_seconds"] else None)
    print("  occupancy", occ, "cmds", ncmd)
    print("  dz med-of-q50 %.4f q90 %.4f q99 %.4f max %.4f ood %.4f droll %.4f" % (
        np.median(q50s), np.median(q90s), np.median(q99s), max(mx), ood_n / ncmd, droll / ncmd))
    print("  g7", r["result"].get("g7_reference_only"))
print("command_statistics numeric values", nvals, "negatives", neg)
