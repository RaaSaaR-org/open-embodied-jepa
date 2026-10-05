"""Scale probe of Stage O's GPU featurisation at the full corpus size (R17.37).

Development only; nothing in it is read as a result. It never reads the sealed corpus: it writes
a synthetic corpus of the sealed corpus's size (1 995 roots: 1 495 train, 250 val, 250 gate) with
uniform-noise frames under the labels 900000-901999 (outside every seed range; nothing is
simulated), and then runs the runner's own ``featurise`` stage on it, in this process, with
``--debug`` (the stage code is the real one: preflight, G-memory at 12 GiB PSS, the 3 600 s cap,
``gpu_guard``, the real DINOv2 encoder on CUDA under strict determinism, the re-mapped stores and
G-anchor). Around it, it samples the process tree's PSS and this process's GPU memory
(``nvidia-smi``), and reads torch's peak allocation at the end. Noise frames compress worse than
rendered ones (about 2.4 MB a root against about 0.61 MB), so reading and hashing the roots is
slower than on the real corpus; the encoder's cost does not depend on the content.

Run from a clean worktree root (the ``featurise`` step through ``scripts/gpu_run.sh --wait``):

    uv run --no-sync python scripts/probe_task077_featurise.py corpus --scratch <new folder>
    uv run --no-sync python scripts/run_lewm_c1m_v2.py tests --output outputs/<tests>
    scripts/gpu_run.sh --wait --who oej:task077-fscale -- uv run --no-sync python \\
        scripts/probe_task077_featurise.py featurise --scratch <folder> \\
        --output outputs/<new> --tests-record outputs/<tests>/report.json

The synthetic corpus and the synthetic features are removed afterwards (``--keep`` keeps them);
the runner's report and this probe's report stay in ``--output``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("_run", ROOT / "scripts/run_lewm_c1m_v2.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)  # sets the declared thread environment before NumPy loads
lm, hz, off, np = run.lm, run.hz, run.off, run.np
GIB = 2**30
LABELS = tuple(range(900000, 902000))  # labels only: nothing is simulated for them
EXCLUDED = 5  # the sealed corpus excluded 5 train roots (R17.35); the probe drops 5 train labels


def build_corpus(scratch: Path) -> dict:
    """1 995 synthetic roots in the sealed corpus's split sizes, sealed like the real one."""
    if scratch.exists():
        raise FileExistsError(f"refusing to overwrite {scratch}")
    corpus = scratch / "corpus"
    corpus.mkdir(parents=True)
    started = time.monotonic()
    split = lm.corpus_split(LABELS)
    dropped = set(split["train"][:EXCLUDED])
    entries = {}
    for label in LABELS:
        if label in dropped:
            continue
        rng = np.random.default_rng(label)
        frames = rng.integers(0, 256, (lm.N_FRAMES, 112, 112, 3), dtype=np.uint8)
        entries[label] = off.write_root(
            corpus,
            label,
            {
                "frames": frames,
                "commands": rng.standard_normal((lm.N_COMMANDS, 14)).astype(np.float32),
                "plate": rng.standard_normal((lm.N_FRAMES, 2)),
                "palm": rng.standard_normal((lm.N_FRAMES, 2)),
                "hidden_r": rng.integers(0, 256, (112, 112, 3), dtype=np.uint8),
                "target": rng.standard_normal(2),
                "state405": rng.standard_normal(16),
                "apple_estimate": rng.standard_normal(2),
                "last_grasp": rng.standard_normal(14),
            },
        )
    sealed = off.seal_corpus(corpus, entries, split, {"synthetic": True, "labels": "900000-901999"})
    kept = {k: len([s for s in v if s in entries]) for k, v in split.items()}
    size = sum(p.stat().st_size for p in corpus.glob("*.npz"))
    return {
        "revision": hz.revision(),
        "dirty": bool(hz.tracked_tree_dirty()),
        "roots": len(entries),
        "kept_per_split": kept,
        "bytes": int(size),
        "seconds": time.monotonic() - started,
        "sealed": sealed,
    }


class Sampler(threading.Thread):
    """The process tree's PSS (0.5 s) and this process's GPU memory from nvidia-smi (1 s)."""

    def __init__(self):
        super().__init__(daemon=True)
        from embodied_jepa.run_guards import process_tree_memory

        self.measure = process_tree_memory
        self.pid = os.getpid()
        self.peak_pss = 0.0
        self.peak_gpu_mib = 0
        self.peak_gpu_total_used_mib = 0
        self.samples = 0
        self.stopped = threading.Event()

    def gpu(self) -> tuple[int, int]:
        mine = total = 0
        try:
            apps = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in apps.strip().splitlines():
                pid, used = (x.strip() for x in line.split(","))
                if int(pid) == self.pid:
                    mine = int(used)
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            total = int(used.strip().splitlines()[0])
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return mine, total

    def run(self):
        tick = 0
        while not self.stopped.is_set():
            self.peak_pss = max(self.peak_pss, self.measure()["pss"] / GIB)
            if tick % 2 == 0:
                mine, total = self.gpu()
                self.peak_gpu_mib = max(self.peak_gpu_mib, mine)
                self.peak_gpu_total_used_mib = max(self.peak_gpu_total_used_mib, total)
            self.samples += 1
            tick += 1
            self.stopped.wait(0.5)


def featurise(args) -> dict:
    import torch

    scratch, output = Path(args.scratch), Path(args.output)
    corpus = scratch / "corpus"
    if not (corpus / "manifest.json").is_file():
        raise FileNotFoundError(f"no synthetic corpus at {corpus}")
    built = json.loads((scratch / "corpus_report.json").read_text())
    probe = {
        "revision": hz.revision(),
        "dirty": bool(hz.tracked_tree_dirty()),
        "corpus": built,
        "gpu_run_locked": os.environ.get("GPU_RUN_LOCKED"),
        "load_at_start": list(os.getloadavg()),
    }
    sampler = Sampler()
    sampler.start()
    started = time.monotonic()
    argv = [
        "featurise",
        "--debug",
        "--output",
        str(output / "stage"),
        "--corpus",
        str(corpus),
        "--tests-record",
        str(args.tests_record),
    ]
    report = run.run(run.build_parser().parse_args(argv))
    wall = time.monotonic() - started
    sampler.stopped.set()
    sampler.join()
    fields = report.get("fields", {})
    feat = fields.get("featurisation") or {}
    cap = lm.CAPS_SECONDS["O_featurisation"]
    probe |= {
        "runner_outcome": report.get("outcome"),
        "runner_total_seconds": report.get("total_seconds"),
        "runner_memory": report.get("memory"),
        "probe_wall_seconds": wall,
        "featurisation_seconds": feat.get("featurisation_seconds"),
        "featurise_corpus_seconds": feat.get("seconds"),
        "anchor": fields.get("anchor"),
        "roots": feat.get("roots"),
        "torch_max_memory_allocated_gib": torch.cuda.max_memory_allocated() / GIB
        if torch.cuda.is_available()
        else None,
        "torch_max_memory_reserved_gib": torch.cuda.max_memory_reserved() / GIB
        if torch.cuda.is_available()
        else None,
        "nvidia_smi_peak_process_gib": sampler.peak_gpu_mib / 1024,
        "nvidia_smi_peak_device_used_gib": sampler.peak_gpu_total_used_mib / 1024,
        "sampler_peak_tree_pss_gib": sampler.peak_pss,
        "sampler_samples": sampler.samples,
        "features_bytes": int(
            sum(p.stat().st_size for p in (output / "stage" / "features").glob("*"))
        )
        if (output / "stage" / "features").exists()
        else None,
        "load_at_end": list(os.getloadavg()),
    }
    worst = report.get("total_seconds") or wall
    probe["caps"] = {
        "cap_O_featurisation_s": cap,
        "stage_seconds": worst,
        "cap_over_worst": cap / worst if worst else None,
        "rule_min": lm.CAP_FACTOR_MIN,
        "memory_ceiling_gib": lm.MEMORY["ceiling_gib"],
        "peak_tree_pss_gib": max(
            sampler.peak_pss, (report.get("memory") or {}).get("peak_tree_pss_gib") or 0.0
        ),
        "gpu_min_free_gib": lm.GPU["min_free_gib"],
        "note": "one run on synthetic noise frames at the sealed corpus's size (1 995 roots); "
        "the stage's seconds include preflight (G-tests are read from --tests-record) and "
        "G-anchor; GPU memory is this process's, against the 8 GiB the guard requires free",
    }
    if not args.keep:
        shutil.rmtree(output / "stage" / "features", ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        probe["synthetic_removed"] = True
    return probe


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("corpus", "featurise"))
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--output")
    parser.add_argument("--tests-record")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.step == "corpus":
        out = build_corpus(Path(args.scratch))
        (Path(args.scratch) / "corpus_report.json").write_text(json.dumps(out, indent=1))
    else:
        if not args.output or not args.tests_record:
            parser.error("featurise needs --output and --tests-record")
        output = Path(args.output)
        output.mkdir(parents=True)  # refuses an existing folder
        out = featurise(args)
        (output / "probe.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
