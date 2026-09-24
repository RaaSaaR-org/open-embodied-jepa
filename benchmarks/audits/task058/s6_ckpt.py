import hashlib
import json

REPO = __import__("os").environ.get("JEPA_ROOT", ".")
WT = REPO
m = json.load(open(WT + "/benchmarks/manifests/apple-policy-v1.json"))
text = json.dumps(m)
for a in ("a0", "a1", "a2", "a3"):
    r = json.load(open(f"{REPO}/outputs/task056-cohort-d/{a}.json"))
    h = hashlib.sha256(open(f"{REPO}/checkpoints/task056-policy-v1/{a}.pt", "rb").read()).hexdigest()
    print(a, r["arm"], "report==file", r["checkpoint_sha256"] == h, "in manifest", h in text,
          r["checkpoint"].split("/")[-2:], r["checkpoint_arm_metadata"], r["device"], r["max_steps"])
