"""TASK-074 run-3 descriptive O1/O2 (NOT a gate; N-7412 unsaturated; no row is read).

Runs from a worktree at main HEAD (>= 5e53ef3) with the frozen code paths of
scripts/run_lewm_planner_v2.py (featurise_sources, train_context, offline_gates) on run-3's
existing checkpoints and saved R_off. Writes only to OUT (a new directory)."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["WT"]).resolve()
RUN3 = Path(os.environ["RUN3"]).resolve()
OUT = Path(os.environ["OUT"]).resolve()
sys.argv = [sys.argv[0]]
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_lewm_planner_v2.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
np = R.np
lp, fm, devices = R.lp, R.fm, R.devices

LABEL = "descriptive, not a gate, N-7412 unsaturated; no row is read from these numbers"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


class Clock:
    def check(self, _label):
        pass


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    import torch

    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import lewm_planner_v2_offline as off
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa import token_dynamics as td
    from embodied_jepa.models.frozen_tokens import frozen_token_model
    from embodied_jepa.models.latent_critic import RidgeReadout

    started = time.monotonic()
    run3_report_path = RUN3 / "outputs/task074-train/run-3/report.json"
    run3_sha = sha_file(run3_report_path)
    run3 = json.loads(run3_report_path.read_text())
    st = run3["stages"]
    rec = {
        "label": LABEL,
        "task": "TASK-074",
        "source_report": str(run3_report_path),
        "source_report_sha256": run3_sha,
        "source_outcome": run3["outcome"],
        "revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        .decode()
        .strip(),
        "tree_dirty": bool(
            subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "-uno"])
        ),
        "frozen_sha256": lp.frozen_sha256(),
        "load_average_at_start": list(os.getloadavg()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": {},
    }
    devices.require("cuda", strict=True)
    torch.set_num_threads(6)
    rec["determinism"] = devices.determinism_state()

    # checkpoints: file sha256 must equal run-3's records
    ckpts = {}
    for name, m in st["models_u80000"].items():
        p = RUN3 / m["checkpoint"]
        got = sha_file(p)
        if got != m["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint sha mismatch {name}")
        ckpts[name] = p
    rec["checks"]["checkpoint_sha256_match"] = True

    with np.load(RUN3 / "outputs/task074-train/run-3/r_off.npz") as data:
        r_off = RidgeReadout.from_state({k: data[k] for k in data.files})
    if r_off.sha256() != st["readouts"]["r_off"]["sha256"]:
        raise SystemExit("r_off sha mismatch")
    rec["checks"]["r_off_sha256_match"] = True

    corpus = ROOT / "data/apple-far-shift-v2"
    corpus_sha = run3["upstream"]["corpus"]["manifest_sha256"]
    reader = rt2.CorpusReader(corpus, corpus_sha, splits=lp.READ_SPLITS)
    evidence = Path(run3["evidence"]["root"])
    look = rt2.CorpusReader(
        evidence / fm.EVIDENCE["corpus"], fm.EVIDENCE["corpus_manifest_sha256"],
        splits=("train", "val"),
    )
    encoder = pe.load_pretrained()
    sources = [("far", reader, s) for s in lp.READ_SPLITS] + [
        ("look", look, s) for s in ("train", "val")
    ]
    table, plate_rows, anchor_frames, anchor_rows = R.featurise_sources(sources, encoder, Clock())
    del plate_rows, anchor_frames
    rec["features"] = {"frames": int(table.size), "roots": len(table.roots)}
    rec["checks"]["features_match_run3"] = (
        int(table.size) == st["features"]["frames"] and len(table.roots) == st["features"]["roots"]
    )
    idx = {
        s: [i for i, r in enumerate(table.roots) if r["split"] == s]
        for s in ("train", "val", "look-train", "look-val")
    }
    starts, owners = table.windows(idx["val"], start_steps=set(lp.DECISION_STEPS))
    base = off.o2_statistics({}, {}, r_off, table, starts, owners)["baselines"]
    rec["checks"]["baselines_match_run3"] = base == st["blind_baselines"]
    rec["baselines"] = base
    schema = rt2.make_robot().state_schema
    ctx = R.train_context(table, idx, schema, {"corpus_manifest_sha256": corpus_sha})

    def load(p):
        m = frozen_token_model(td.BACKEND)(schema, device="cuda", seed=0, config=td.MODEL_CONFIG)
        m.load(p)
        m.eval()
        return m

    w, n, vc = {}, {}, {}
    for s in lp.MODEL_SEEDS:
        w[s] = load(ckpts[f"W-{s}"])
        n[s] = load(ckpts[f"N-{s}"])
        for arm, m in (("W", w[s]), ("N", n[s])):
            v = off.val_criterion(m, table, ctx["val_starts"], ctx["scale"], arm)
            want = st["models_u80000"][f"{arm}-{s}"]["selected_val_criterion"]
            vc[f"{arm}-{s}"] = {"recomputed": v, "run3_selected": want, "equal": v == want}
    rec["checks"]["val_criterion_reproduces_selected"] = vc
    rec["selected_updates"] = {k: v["selected_update"] for k, v in st["models_u80000"].items()}
    gates = R.offline_gates(ctx, table, idx, w, n, r_off)
    rec["o1"] = {
        str(s): v["gate"]
        | {"statistics": {f"{k[0]}@h{k[1]}": v2 for k, v2 in v["statistics"].items()}}
        for s, v in gates["o1"].items()
    }
    rec["o2"] = {str(s): gates["o2"][s] for s in lp.MODEL_SEEDS}
    rec["o2_baselines"] = gates["o2"]["baselines"]
    rec["test_split_decoded"] = bool(reader.test_split_decoded)
    rec["seconds"] = time.monotonic() - started
    rec["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = OUT / "descriptive.json"
    path.write_text(json.dumps(clean(rec), indent=2, sort_keys=True))
    print("written", path, sha_file(path))


main()
