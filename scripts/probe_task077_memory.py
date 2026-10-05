"""Memory probe of Stage C's seed preparation and Stage O's featurisation (R17.30-R17.31).

Development only; nothing in it is read as a result. It simulates debug seeds 66900-66999 only
(no K, D, S or corpus seed), uses no GPU, and writes only to a new folder it is given. Run from
the worktree root:

    MUJOCO_GL=egl uv run --no-sync python scripts/probe_task077_memory.py seedprep outputs/<new>
    uv run --no-sync python scripts/probe_task077_memory.py featurise outputs/<new>

``seedprep``: the runner's own setup (``sim_setup``, debug, real pool, P-3's refitted readout and
the encoder), then

1. per-process PSS after setup (main process against each worker);
2. on the 100 debug seeds' real post-look tokens: the pinned ``XYReadout.predict`` against the
   runner's ``streamed_estimates`` at chunks of 32, 64 and 128 (bitwise), and the pinned
   ``cohort_estimates`` against the runner's, end to end (both render);
3. on 2 000 synthetic token rows (each a debug row times 1 + 1e-3 N(0, 1)): the same comparison;
4. Stage C's seed preparation at 2 000 seeds, ``Cohorts.seeds("corpus")`` itself, once with the
   runner's ``cohort_estimates`` and once with the pinned one (the void's path), each with the
   real pool alive and a stand-in pool that answers each frame task with a copy of a rendered
   debug frame (so nothing is rendered for those 2 000 synthetic seeds; ``reset_of`` and the seed
   list are stand-ins); the per-process PSS peak and NumPy's traced peak of each, and the two
   paths' estimates compared bitwise.

``featurise``: a synthetic 2 000-root corpus (``write_root``, synthetic arrays) featurised by
``featurise_corpus`` on the CPU with a stand-in encoder (an average pool of the input; no weights),
once with the re-mapping (R17.31) and once with one mapping per split (as at ``862d63c``): the
peak PSS of each and the written files compared by sha256.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import tracemalloc
from pathlib import Path

ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("_run", ROOT / "scripts/run_lewm_c1m_v2.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
lm, hz, off, np = run.lm, run.hz, run.off, run.np
GIB = 2**30
SYNTHETIC_SEEDS = tuple(range(900000, 902000))  # labels only: nothing is simulated for them


class Sampler(threading.Thread):
    """Per-process PSS of this process tree every ``interval`` s, with per-phase peaks."""

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        from embodied_jepa.run_guards import process_tree_memory

        self.measure, self.interval = process_tree_memory, interval
        self.main = os.getpid()
        self.phase, self.peaks, self.lock = "start", {}, threading.Lock()
        self.stopped = threading.Event()

    def sample(self):
        per = {}
        both = self.measure(per=per)
        main = per.get(self.main, (0, 0))[1]
        workers = both["pss"] - main
        with self.lock:
            p = self.peaks.setdefault(
                self.phase,
                {"tree_pss_gib": 0.0, "main_pss_gib": 0.0, "workers_pss_gib": 0.0, "samples": 0},
            )
            p["samples"] += 1
            if both["pss"] / GIB > p["tree_pss_gib"]:
                p["tree_pss_gib"] = both["pss"] / GIB
                p["at_tree_peak"] = {
                    "main_pss_gib": main / GIB,
                    "others_pss_gib": workers / GIB,
                    "processes": len(per),
                    "per_process_pss_gib": sorted(
                        (round(v[1] / GIB, 3) for k, v in per.items() if k != self.main),
                        reverse=True,
                    ),
                }
            p["main_pss_gib"] = max(p["main_pss_gib"], main / GIB)
            p["workers_pss_gib"] = max(p["workers_pss_gib"], workers / GIB)
        return both

    def set(self, phase):
        self.sample()
        with self.lock:
            self.phase = phase
        self.sample()

    def run(self):
        while not self.stopped.wait(self.interval):
            self.sample()


def wait_quiet(limit=2.0):
    while True:
        one, five, _ = os.getloadavg()
        if one <= limit and five <= limit:
            return [one, five]
        print(f"waiting for load <= {limit} ({one:.2f}, {five:.2f})", flush=True)
        time.sleep(30)


class StandInPool:
    """Answers each frame task with a copy of a rendered debug frame (nothing is rendered)."""

    def __init__(self, rendered: list[dict]):
        self.rendered = rendered

    def map(self, tasks, cap, what):
        out = []
        for t in tasks:
            src = self.rendered[t["source"]]
            item = dict(src)
            item["frame"] = np.array(src["frame"], copy=True)
            item["seed"] = t["seed"]
            out.append(item)
        return out


def seedprep(out_dir: Path) -> dict:
    from embodied_jepa import first_policy_perception as fpp

    args = run.build_parser().parse_args(
        [
            "corpus",
            "--debug",
            "--output",
            str(out_dir / "unused"),
            "--evidence",
            "/home/huhn/develop/emai/worktrees/task076-evidence",
        ]
    )
    report = {"stages": {}, "revision": hz.revision(), "dirty": bool(hz.tracked_tree_dirty())}
    report["load_start"] = wait_quiet()
    sampler = Sampler()
    sampler.start()
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    sampler.set("setup")
    pool, co = run.sim_setup(report, args, manifest, {})
    sampler.set("after_setup_idle")
    time.sleep(3)
    readout, encoder = co.p_readout, co.encoder
    report["p_readout_rows"] = list(readout.features.shape)
    report["p_readout_features_gib"] = readout.features.nbytes / GIB
    seeds = lm.check_seeds("corpus", tuple(range(66900, 67000)), debug=True)
    resets = {s: lm.reset_of(s) for s in seeds}
    try:
        # 2. the 100 debug seeds' real tokens
        sampler.set("debug_render")
        tasks = [{"kind": "frame", "seed": s, "reset": resets[s]} for s in seeds]
        frames, disagreements = hz.render_majority(pool, tasks, 1800.0, "debug frames")
        tokens = np.concatenate([fpp.featurise(encoder, f["frame"][None]) for f in frames])
        reference = readout.predict(tokens)
        real = {"render_disagreements": len(disagreements), "token_dtype": str(tokens.dtype)}
        for chunk in (32, 64, 128):
            got = run.streamed_estimates(
                readout, (tokens[lo : lo + chunk] for lo in range(0, len(tokens), chunk))
            )
            real[f"chunk_{chunk}_bit_identical"] = bool(np.array_equal(got, reference))
            real[f"chunk_{chunk}_max_abs_diff"] = float(np.max(np.abs(got - reference)))
        pinned = hz.cohort_estimates(pool, readout, encoder, seeds, resets, 1800.0)
        mine = run.cohort_estimates(pool, readout, encoder, seeds, resets, 1800.0, chunk=32)
        same_frames = [s for s in seeds if pinned[s]["frame_sha256"] == mine[s]["frame_sha256"]]
        real["end_to_end"] = {
            "seeds": len(seeds),
            "same_frame": len(same_frames),
            "estimates_bit_identical_on_same_frame": all(
                pinned[s]["estimates"] == mine[s]["estimates"] for s in same_frames
            ),
            "estimates_bit_identical_all": all(
                pinned[s]["estimates"] == mine[s]["estimates"] for s in seeds
            ),
        }
        report["debug_100"] = real

        # 3. 2 000 synthetic token rows
        sampler.set("synthetic_tokens")
        rng = np.random.default_rng(0)
        idx = np.arange(len(SYNTHETIC_SEEDS)) % len(tokens)
        synthetic = tokens[idx] * (1.0 + 1e-3 * rng.standard_normal((len(idx), tokens.shape[1])))
        reference = readout.predict(synthetic)
        syn = {"rows": len(synthetic)}
        for chunk in (32, 128, 256):
            got = run.streamed_estimates(
                readout, (synthetic[lo : lo + chunk] for lo in range(0, len(synthetic), chunk))
            )
            syn[f"chunk_{chunk}_bit_identical"] = bool(np.array_equal(got, reference))
            syn[f"chunk_{chunk}_max_abs_diff"] = float(np.max(np.abs(got - reference)))
        report["synthetic_2000"] = syn
        del synthetic, reference, got, tokens

        # 4. Stage C's seed preparation at 2 000 seeds
        stand_in = StandInPool(frames)
        source = {s: i % len(frames) for i, s in enumerate(SYNTHETIC_SEEDS)}
        debug_reset = {s: resets[seeds[source[s]]] | {"source": source[s]} for s in source}
        saved = (lm.seeds_of, lm.check_seeds, lm.reset_of, run.cohort_estimates)
        lm.seeds_of = lambda role, debug=False: SYNTHETIC_SEEDS
        lm.check_seeds = lambda role, s, debug=False: tuple(s)
        lm.reset_of = lambda s: debug_reset[s]
        original_render = hz.render_majority

        def render_with_source(_pool, tasks, cap, what):
            for t in tasks:
                t["source"] = t["reset"]["source"]
            return original_render(stand_in, tasks, cap, what)

        hz.render_majority = render_with_source
        results = {}
        try:
            for name, fn in (("fixed", saved[3]), ("pinned_unchunked", hz.cohort_estimates)):
                wait_quiet()
                run.cohort_estimates = fn
                cohort = run.Cohorts(args, pool, readout, encoder, {})
                time.sleep(2)
                sampler.set(f"seedprep_2000_{name}")
                tracemalloc.start()
                t = time.monotonic()
                cohort.seeds("corpus")
                seconds = time.monotonic() - t
                traced = tracemalloc.get_traced_memory()[1] / GIB
                tracemalloc.stop()
                sampler.set(f"after_{name}")
                results[name] = {
                    "seconds": seconds,
                    "numpy_traced_peak_gib": traced,
                    "estimates": [cohort.est[s]["estimates"] for s in SYNTHETIC_SEEDS],
                }
                del cohort
                time.sleep(3)
        finally:
            lm.seeds_of, lm.check_seeds, lm.reset_of, run.cohort_estimates = saved
            hz.render_majority = original_render
        report["seedprep_2000"] = {
            name: {k: v for k, v in r.items() if k != "estimates"} for name, r in results.items()
        }
        report["seedprep_2000"]["estimates_bit_identical"] = (
            results["fixed"]["estimates"] == results["pinned_unchunked"]["estimates"]
        )
    finally:
        report["pool_close"] = pool.close()
        sampler.stopped.set()
        report["phases"] = sampler.peaks
        report["load_end"] = os.getloadavg()
        report.pop("_run1", None)
    return report


class StandInEncoder:
    """Deterministic [B, 257, 384] 'tokens' from the input (a 14 x 14 average pool); no weights."""

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values):
        import types

        import torch
        import torch.nn.functional as F

        p = F.avg_pool2d(pixel_values, 14)  # [B, 3, 16, 16]
        t = p.flatten(2).transpose(1, 2).repeat(1, 1, 128)  # [B, 256, 384]
        cls = t.new_zeros((len(t), 1, 384))
        return types.SimpleNamespace(last_hidden_state=torch.cat([cls, t], dim=1))


def synthetic_root(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base = (np.arange(112 * 112 * 3, dtype=np.int64).reshape(112, 112, 3) + seed) % 251
    frames = np.stack([(base + t) % 251 for t in range(lm.N_FRAMES)]).astype(np.uint8)
    return {
        "frames": frames,
        "commands": rng.standard_normal((lm.N_COMMANDS, 14)).astype(np.float32),
        "plate": rng.standard_normal((lm.N_FRAMES, 2)),
        "palm": rng.standard_normal((lm.N_FRAMES, 2)),
        "hidden_r": frames[-1],
        "target": rng.standard_normal(2),
        "state405": rng.standard_normal(16),
        "apple_estimate": rng.standard_normal(2),
        "last_grasp": rng.standard_normal(14),
    }


def featurise(out_dir: Path) -> dict:
    import torch

    torch.set_num_threads(6)
    report = {"revision": hz.revision(), "dirty": bool(hz.tracked_tree_dirty())}
    report["load_start"] = wait_quiet()
    corpus = out_dir / "corpus"
    corpus.mkdir()
    seeds = SYNTHETIC_SEEDS
    entries = {s: off.write_root(corpus, s, synthetic_root(s)) for s in seeds}
    split = lm.corpus_split(seeds)
    off.seal_corpus(corpus, entries, split, {"synthetic": True})
    manifest = off.open_corpus(corpus, None)
    sampler = Sampler(interval=0.2)
    sampler.start()
    results = {}
    for name, remap in (("remapped", off.FEATURE_REMAP_ROOTS), ("one_mapping", None)):
        wait_quiet()
        sampler.set(f"featurise_{name}")
        t = time.monotonic()
        result = off.featurise_corpus(
            corpus,
            manifest,
            out_dir / name,
            device="cpu",
            encoder=StandInEncoder(),
            remap_every=remap,
        )
        sampler.set(f"after_{name}")
        results[name] = {"seconds": time.monotonic() - t, "files_sha256": result["files_sha256"]}
    sampler.stopped.set()
    report["featurise"] = {name: {"seconds": r["seconds"]} for name, r in results.items()} | {
        "files_identical": results["remapped"]["files_sha256"]
        == results["one_mapping"]["files_sha256"],
        "remap_every": off.FEATURE_REMAP_ROOTS,
    }
    report["phases"] = sampler.peaks
    report["load_end"] = os.getloadavg()
    for name in ("corpus", "remapped", "one_mapping"):
        shutil.rmtree(out_dir / name, ignore_errors=True)  # synthetic only
    return report


def main():
    mode, out_dir = sys.argv[1], ROOT / sys.argv[2]
    out_dir.mkdir(parents=True)  # refuses an existing folder
    report = {"seedprep": seedprep, "featurise": featurise}[mode](out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
