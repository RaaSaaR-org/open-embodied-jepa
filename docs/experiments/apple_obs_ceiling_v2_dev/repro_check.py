"""Development check before the freeze: does the readouts stage's G-repro-off path, under the
stage's GPU cap and strict determinism, reproduce TASK-074's published R_off numbers exactly on
the sealed apple-far-shift-v2 reference frames? It reads only what TASK-074 already read and
published; no new view, no new readout."""
import json, sys, time
from pathlib import Path
W = Path("/home/huhn/develop/emai/worktrees/task075-prereg")
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("runner", W / "scripts/run_obs_ceiling_v2.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
from embodied_jepa import devices, first_policy_v2_runtime as rt2, obs_ceiling_v2 as oc
from embodied_jepa import pretrained_encoder as pe
devices.require("cuda", strict=True)
report = {}
runner.gpu_guard(report)
reader = rt2.CorpusReader(Path("/home/huhn/develop/emai/worktrees/task074-run/data/apple-far-shift-v2"),
                          oc.SOURCE_CORPUS["manifest_sha256"], splits=("train", "val"))
clock = runner.R65.Clock(3600)
started = time.time()
rec = runner.repro_off(report, reader, pe.load_pretrained(), clock, smoke=True)
runner.gpu_peak(report)
out = {"repro_off": rec, "gpu": report, "test_split_decoded": reader.test_split_decoded,
       "decoded": len(reader.decoded), "seconds": time.time() - started}
print(json.dumps(out, indent=1))
