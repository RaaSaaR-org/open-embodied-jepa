exec(open(__file__.replace("s5_blind3.py", "s5_blind.py")).read().split("res = {}")[0])
# palm position = apple + pma; per-step palm motion within episodes
palm = va["apos"] + va["pma"]
same = np.r_[False, va["ep"][1:] == va["ep"][:-1]]
step = np.r_[np.nan, np.linalg.norm(np.diff(palm, axis=0), axis=1)]
step[~same] = np.nan
for p, n in ((0, "orient"), (1, "descend"), (2, "close")):
    m = vk & (va["phase"] == p) & same
    print(n, "rows", m.sum(), "frac palm step < 1 mm: %.3f" % np.mean(step[m] < 0.001),
          "median |pma| cm %.2f" % (100 * np.median(np.linalg.norm(va["pma"][m], axis=1))))
# analytic blind prior check
rng = np.random.default_rng(0)
u = rng.uniform(-0.03, 0.03, (2_000_000, 2))
print("P0b median norm uniform square cm %.4f" % (100 * np.median(np.linalg.norm(u, axis=1))))
