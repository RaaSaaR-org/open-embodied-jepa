import json

D = __import__("os").environ.get("JEPA_ROOT", ".") + "/checkpoints/task056-policy-v1/"
for a in ("a0", "a1", "a2", "a3"):
    d = json.load(open(D + a + ".run.json"))
    b = [v for v in d["validation"] if v["step"] == d["best_step"]][0]
    print(a, d["status"], d["completed_steps"], d["best_step"], d["best_selection_score"],
          d["budget"], round(d.get("elapsed_seconds", 0), 1),
          d["data"]["train"]["rows_sampled"], d["data"]["val"]["rows_sampled"],
          d["source"]["revision"][:8], d["encoder"])
    print("  perdim", {k: round(v, 4) for k, v in b["per_dimension"].items()},
          "arm", round(b["median_abs_error_arm"], 4), "grasp", round(b["median_abs_error_grasp"], 4))
    print("  std", {k: round(v, 4) for k, v in b["output_std_per_dimension"].items()})
    print("  last", [(v["step"], round(v["median_abs_error"], 5)) for v in d["validation"][-4:]])
    print("  keys", [k for k in d if "second" in k or "clock" in k or "elapsed" in k])
