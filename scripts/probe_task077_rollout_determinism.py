"""TASK-077 development probe: is W's CPU roll-out bit-reproducible on identical input?

Development only (not pinned; it gates nothing; nothing in it is read as a result). It answers
the #146 approval's non-blocking note 2, to be settled before Stage S's GO (R17.43): the 0.6 cm
commit-target gate (R17.38) assumes that "identical frames give identical targets on the CPU".
This probe measures that, across separate processes, under the run's thread settings.

**Inputs (no gated data).**
- The Stage-0 debug chain at ``9e772e0`` (``--smoke``; debug seeds 66900-66999 only): the 405
  frames of debug corpus roots, the debug-trained ``W-66992`` and ``N-66992`` checkpoints, and the
  debug R-plate, R8 and mean latent, each checked against the sha256 its debug report records.
- A random-init model of the real architecture (``lewm_c1m_v2_train.build_model``, torch seed
  66995 in the debug range), with the debug moments, written to ``--scratch``.
- The candidate commands are synthetic but deterministic: each candidate's 60 commands are the
  root's logged executed commands from 405 with the candidate's offset from the plate reading
  added to the first two dimensions. The kinematic stand-in (``place_planner.primitive_chunks``,
  MuJoCo) is not exercised here; the simulator's determinism is G-repro's.

**What each child process does** (a fresh interpreter with the runner's thread environment,
``MKL_DYNAMIC=FALSE``, ``OMP_NUM_THREADS=6``, ``MKL_NUM_THREADS=6``, ``OPENBLAS_NUM_THREADS=16``,
set before NumPy loads, and one torch thread, the closed-loop workers' ``WORKER_TORCH_THREADS``):
for every root and every arm (W, N and L-mean on the debug models; W on the random-init model),
the closed loop's own functions: ``lewm_c1m_v2_runtime.encode`` (DINOv2 on the CPU), R-plate's
reading, the 147-candidate grid (``lewm_next_c1.from_box``), ``rollout_plates`` and
``choose_from_grid`` with the frozen refinement tolerance (τ_commit/4 = 0.25 cm). It records the
sha256 of every array along the way (tokens, the pooled latent, the reading, every roll-out's
plates, the commit target) and repeats the whole pass in the same process.

The parent runs ``--sequential`` children one after another, then ``--concurrent`` children at
once (as Stage S's 4 closed-loop workers run), and reports, per root and arm, whether every
digest is the same in every process and pass. Outcome ``IDENTICAL`` or ``DIFFERENT``.

Usage::

    uv run --no-sync python scripts/probe_task077_rollout_determinism.py run \\
      --output outputs/task077-rdet-1 --scratch <scratch> \\
      --smoke /home/huhn/develop/emai/worktrees/task077-stage0/outputs/task077-smoke-9e772e0
"""

from __future__ import annotations

import os

THREAD_ENV = {  # the runner's pinned environment (G-threads, R17.29), before NumPy loads
    "MKL_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "16",
}
os.environ.update(THREAD_ENV)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RANDOM_INIT_SEED = 66995  # debug range 66900-66999; unused elsewhere
DEBUG_ROOTS = (66953, 66965, 66967, 66975)  # the debug chain's val roots (debug seeds)
ARMS = (
    ("W", "W-debug", "own"),
    ("N", "N-debug", "own"),
    ("L-mean", "W-debug", "mean"),
    ("W", "W-random-init", "own"),
)


def sha(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


# ----- the child ----------------------------------------------------------------------------------
def child(config: dict) -> dict:
    import torch

    from embodied_jepa import lewm_c1m_v2 as lm
    from embodied_jepa import lewm_c1m_v2_offline as off
    from embodied_jepa import lewm_c1m_v2_runtime as wrt
    from embodied_jepa import lewm_c1m_v2_train as tr
    from embodied_jepa import lewm_next_c1 as c1
    from embodied_jepa.models.latent_critic import RidgeReadout

    torch.set_num_threads(int(lm.WORKER_TORCH_THREADS))
    models = {
        name: tr.load_model(
            spec["path"],
            seed=int(spec["seed"]),
            metadata=dict(spec["metadata"]),
            expected_sha256=spec["sha256"],
        )
        for name, spec in config["models"].items()
    }
    readouts = {}
    for name in ("r_plate", "r8"):
        with np.load(config[name]["path"]) as data:
            readouts[name] = RidgeReadout.from_state({k: data[k] for k in data.files})
        if readouts[name].sha256() != config[name]["sha256"]:
            raise SystemExit(f"{name} differs from its recorded sha256")
    mean = np.asarray(np.load(config["mean_latent"]["path"]), np.float32)
    if (
        hashlib.sha256(np.ascontiguousarray(mean).tobytes()).hexdigest()
        != (config["mean_latent"]["sha256"])
    ):
        raise SystemExit("the mean latent differs from its recorded sha256")
    tolerance = lm.REFINE_FRACTION_OF_TAU * float(lm.K0_MEASURED["tau_commit_cm"]) / 100.0
    passes = []
    for _rep in range(int(config["passes"])):
        started = time.perf_counter()
        out = {}
        for seed in config["roots"]:
            root = off.load_root(Path(config["corpus"]), seed)
            k405 = lm.COMMIT_STEP - lm.FRAME_STEPS[0]
            tokens, own = wrt.encode(root["frames"][k405])
            p_hat = np.asarray(readouts["r_plate"].predict(tokens), np.float64).reshape(2)
            h = np.asarray(root["palm"][k405], np.float64)
            targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in lm.GRID])
            base = np.asarray(root["commands"][k405 : k405 + lm.HORIZON], np.float32)

            def chunk(g, base=base, p_hat=p_hat):
                c = base.copy()
                c[:, :2] += (10.0 * (np.asarray(g) - p_hat)).astype(np.float32)
                return np.clip(c, -1.0, 1.0)

            chunks = np.stack([chunk(t) for t in targets])
            feasible = np.ones(len(targets), bool)
            row = {
                "tokens": sha(tokens),
                "own_latent": sha(own),
                "p_hat": sha(p_hat),
                "chunks": sha(chunks),
            }
            for arm, model_name, start_kind in ARMS:
                model = models[model_name]
                start = own if start_kind == "own" else mean
                plates_seen = []

                def predict(commands, model=model, start=start, arm=arm, seen=plates_seen):
                    p = wrt.rollout_plates(model, start, commands, readouts["r8"], zero=arm == "N")
                    seen.append(sha(p))
                    return p

                g, log = wrt.choose_from_grid(
                    arm,
                    p_hat,
                    h,
                    targets,
                    chunks,
                    feasible,
                    predict,
                    lambda g: (chunk(g), True),
                    tolerance_m=tolerance,
                    seed=int(seed),
                )
                row[f"{arm}/{model_name}"] = {
                    "target_sha256": sha(np.asarray(g, np.float64)),
                    "target": np.asarray(g, np.float64).tolist(),
                    "rollouts_sha256": hashlib.sha256("".join(plates_seen).encode()).hexdigest(),
                    "rollouts": int(log["rollouts"]),
                    "grid_index": int(log["grid_index"]),
                    "converged": bool(log["converged"]),
                    "clipped": bool(log["clipped"]),
                    "refinement_steps": len(log["iterations"]),
                }
            out[str(seed)] = row
        passes.append({"rows": out, "seconds": time.perf_counter() - started})
    return {
        "pid": os.getpid(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "passes": passes,
    }


# ----- the parent ---------------------------------------------------------------------------------
def write_random_init(scratch: Path, moments_path: str, moments_sha: str) -> dict:
    from embodied_jepa import lewm_c1m_v2_offline as off
    from embodied_jepa import lewm_c1m_v2_train as tr

    with np.load(moments_path) as data:
        mean, std = data["mean"], data["std"]
    if off.moments_sha256(mean, std) != moments_sha:
        raise SystemExit("the debug moments differ from their recorded sha256")
    metadata = tr.model_metadata(
        arm="W", seed=RANDOM_INIT_SEED, corpus_sha256="random-init", moments_sha256=moments_sha
    )
    model = tr.build_model(seed=RANDOM_INIT_SEED, device="cpu", metadata=metadata)
    model.fit_frozen_feature_normalization(mean, std, training_episode_ids=("random-init",))
    path = scratch / f"W-random-init-{RANDOM_INIT_SEED}.pt"
    model.save(path)
    return {
        "path": str(path),
        "sha256": tr.sha256_file(path),
        "seed": RANDOM_INIT_SEED,
        "metadata": metadata,
    }


def config_from_smoke(smoke: Path, scratch: Path, passes: int) -> dict:
    readouts = json.loads((smoke / "readouts" / "report.json").read_text())
    if not str(readouts.get("outcome", "")).endswith("-DEBUG") or not readouts.get("debug"):
        raise SystemExit("the smoke readouts report is not a debug report")
    fits = readouts["fields"]["fits"]

    def fit(name):
        return {"path": str(smoke / "readouts" / "fits" / Path(fits[name]["path"]).name)} | {
            "sha256": fits[name]["sha256"]
        }

    models = {}
    for name in ("W", "N"):
        report = json.loads((smoke / f"train-{name}-66992" / "report.json").read_text())
        if not report.get("debug"):
            raise SystemExit("a smoke model report is not a debug report")
        record = report["fields"]["record"]
        models[f"{name}-debug"] = {
            "path": str(smoke / f"train-{name}-66992" / Path(record["checkpoint"]).name),
            "sha256": record["checkpoint_sha256"],
            "seed": record["seed"],
            "metadata": record["metadata"],
        }
    moments = fit("moments")
    models["W-random-init"] = write_random_init(scratch, moments["path"], moments["sha256"])
    return {
        "corpus": str(smoke / "corpus" / "corpus"),
        "roots": list(DEBUG_ROOTS),
        "models": models,
        "r_plate": fit("r_plate"),
        "r8": fit("r8"),
        "mean_latent": fit("mean_latent"),
        "passes": int(passes),
    }


def spawn(config_path: Path, out_path: Path) -> subprocess.Popen:
    env = dict(os.environ) | THREAD_ENV | {"CUDA_VISIBLE_DEVICES": ""}
    return subprocess.Popen(
        [sys.executable, __file__, "child", "--config", str(config_path), "--out", str(out_path)],
        env=env,
    )


def compare(children: list[dict]) -> dict:
    """Per root and arm: is every digest the same in every process and pass?"""
    rows = [p["rows"] for c in children for p in c["passes"]]
    per, different = {}, []
    for seed, first in rows[0].items():
        for key, value in first.items():
            if isinstance(value, dict):
                digests = {
                    (r[seed][key]["target_sha256"], r[seed][key]["rollouts_sha256"]) for r in rows
                }
                targets = np.asarray([r[seed][key]["target"] for r in rows])
                spread = float(np.max(np.linalg.norm(targets - targets[0], axis=1)))
                same = len(digests) == 1
                per[f"{seed}/{key}"] = {
                    "identical": same,
                    "distinct_digests": len(digests),
                    "max_target_difference_m": spread,
                    "converged": first[key]["converged"],
                    "clipped": first[key]["clipped"],
                    "rollouts": first[key]["rollouts"],
                }
            else:
                same = len({r[seed][key] for r in rows}) == 1
                per[f"{seed}/{key}"] = {"identical": same}
            if not same:
                different.append(f"{seed}/{key}")
    return {"comparisons": per, "different": different, "observations": len(rows)}


def run(args) -> int:
    output = Path(args.output)
    scratch = Path(args.scratch)
    for path in (output, scratch):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    revision = git("rev-parse", "HEAD")
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    started = time.time()
    load_start = list(os.getloadavg())
    config = config_from_smoke(Path(args.smoke), scratch, args.passes)
    config_path = scratch / "config.json"
    config_path.write_text(json.dumps(config, indent=1))
    results, runs = [], []
    for k in range(int(args.sequential)):
        out = scratch / f"seq-{k}.json"
        t0 = time.time()
        code = spawn(config_path, out).wait()
        runs.append(
            {"name": f"seq-{k}", "mode": "sequential", "exit": code, "seconds": time.time() - t0}
        )
        if code:
            raise SystemExit(f"child seq-{k} failed ({code})")
        results.append(json.loads(out.read_text()) | {"name": f"seq-{k}"})
    t0 = time.time()
    procs = [(f"conc-{k}", scratch / f"conc-{k}.json") for k in range(int(args.concurrent))]
    handles = [(name, out, spawn(config_path, out)) for name, out in procs]
    for name, out, handle in handles:
        code = handle.wait()
        runs.append({"name": name, "mode": "concurrent", "exit": code})
        if code:
            raise SystemExit(f"child {name} failed ({code})")
        results.append(json.loads(out.read_text()) | {"name": name})
    concurrent_seconds = time.time() - t0
    verdict = compare(results)
    report = {
        "probe": "task077-rollout-determinism",
        "task": "TASK-077",
        "development_only": True,
        "revision": revision,
        "tracked_tree_dirty": dirty,
        "revision_at_end": git("rev-parse", "HEAD"),
        "tracked_tree_dirty_at_end": bool(git("status", "--porcelain", "--untracked-files=no")),
        "smoke": str(Path(args.smoke).resolve()),
        "inputs": {
            "roots": list(DEBUG_ROOTS),
            "models": {
                k: {"sha256": v["sha256"], "seed": v["seed"]} for k, v in config["models"].items()
            },
            "r_plate_sha256": config["r_plate"]["sha256"],
            "r8_sha256": config["r8"]["sha256"],
            "mean_latent_sha256": config["mean_latent"]["sha256"],
            "candidate_commands": "synthetic and deterministic: the root's logged commands from "
            "405 with 10 x (candidate - reading) added to the first two dimensions, clipped",
        },
        "arms": [f"{a}/{m}" for a, m, _s in ARMS],
        "thread_env": THREAD_ENV,
        "children": [
            {
                "name": c["name"],
                "pid": c["pid"],
                "torch_threads": c["torch_threads"],
                "thread_env": c["thread_env"],
                "pass_seconds": [p["seconds"] for p in c["passes"]],
            }
            for c in results
        ],
        "runs": runs,
        "concurrent_seconds": concurrent_seconds,
        "torch_version": results[0]["torch_version"],
        "python": platform.python_version(),
        "machine": platform.processor() or platform.machine(),
        "load_average_at_start": load_start,
        "load_average_at_end": list(os.getloadavg()),
        "seconds": time.time() - started,
        "outcome": "IDENTICAL" if not verdict["different"] else "DIFFERENT",
        **verdict,
    }
    (output / "report.json").write_text(json.dumps(report, indent=1))
    if not args.keep_scratch:
        for path in sorted(scratch.iterdir()):
            path.unlink()
        scratch.rmdir()
        report["scratch_removed"] = True
        (output / "report.json").write_text(json.dumps(report, indent=1))
    print(
        f"outcome {report['outcome']}: {verdict['observations']} observations, "
        f"{len(verdict['different'])} different"
    )
    return 0 if not verdict["different"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    r.add_argument("--scratch", required=True)
    r.add_argument("--smoke", required=True)
    r.add_argument("--passes", type=int, default=2)
    r.add_argument("--sequential", type=int, default=3)
    r.add_argument("--concurrent", type=int, default=4)
    r.add_argument("--keep-scratch", action="store_true")
    c = sub.add_parser("child")
    c.add_argument("--config", required=True)
    c.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "child":
        result = child(json.loads(Path(args.config).read_text()))
        Path(args.out).write_text(json.dumps(result))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
