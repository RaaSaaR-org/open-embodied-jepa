exec(open(__file__.replace("s5_blind2.py", "s5_blind.py")).read().split("res = {}")[0])


def X(d, m):
    return np.c_[d["state"][m], np.ones(m.sum())]


for tgt in ("pma", "apos"):
    W, *_ = np.linalg.lstsq(X(tr, tk), tr[tgt][tk], rcond=None)
    for name, phs in {"close": [2], "approach": [0, 1], "orient": [0]}.items():
        vm = vk & np.isin(va["phase"], phs)
        e = np.linalg.norm(X(va, vm) @ W - va[tgt][vm], axis=1)
        print("all-phase proprio-linear", tgt, name, "%.3f cm" % (100 * np.median(e)))
for name, phs in {"close": [2], "approach": [0, 1]}.items():
    vm = vk & np.isin(va["phase"], phs)
    med = np.median(tr["pma"][tk], 0)
    print("global train median pma", name,
          "%.3f cm" % (100 * np.median(np.linalg.norm(va["pma"][vm] - med, axis=1))))
m0 = tk & (tr["phase"] == 0)
print("orient pma std cm", 100 * tr["pma"][m0].std(0), "apos std", 100 * tr["apos"][m0].std(0))
first = np.r_[True, tr["ep"][1:] != tr["ep"][:-1]]
print("frame0 pma std cm", 100 * tr["pma"][first].std(0), "apos", 100 * tr["apos"][first].std(0))
print("frame0 state std (joint pos, first 29)", np.round(tr["state"][first][:, 15:29].std(0), 4))
